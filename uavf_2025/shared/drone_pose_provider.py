import bisect
import rclpy
import rclpy.node
from rclpy.executors import SingleThreadedExecutor
from .types import Pose, GlobalPosition
from .ensure_ros import ensure_ros
import threading
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix
from mavros_msgs.srv import StreamRate
import rclpy.qos
from pathlib import Path
import time
import json
import logging


class DronePoseProvider(rclpy.node.Node):
    @ensure_ros
    def __init__(
        self, console_logger: logging.Logger, data_log_dir: Path | None = None
    ):
        super().__init__("uavf_2025_drone_pose_provider")

        self.qos_profile = rclpy.qos.QoSProfile(
            depth=1, reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT
        )
        # Set up ROS2 subscription to receive PoseStamped messages

        self.create_subscription(
            PoseStamped,
            "/mavros/local_position/pose",
            self._local_listener_callback,
            qos_profile=self.qos_profile,
        )
        self.create_subscription(
            NavSatFix,
            "/mavros/global_position/global",
            self._global_listener_callback,
            self.qos_profile,
        )
        self.stream_rate_client = self.create_client(
            StreamRate, "/mavros/set_stream_rate"
        )
        self.latest_gps_msg = None
        self.data_log_dir = data_log_dir
        self.console_logger = console_logger

        # Buffer to store recent poses with their timestamps for interpolation/extrapolation
        self.pose_buffer = []
        self.timestamp_buffer = []
        self.buffer_size = 200  # Maximum number of poses in the buffer
        self._testing_time_offset_sec = None  # when testing with ROSBag

        # Custom executor and thread for running the node in a single thread
        self.custom_executor = SingleThreadedExecutor()
        self.custom_executor.add_node(self)
        self.thread = threading.Thread(target=self.spin_thread, daemon=True)
        self.thread.start()

        if not self.stream_rate_client.wait_for_service(timeout_sec=10):
            self.console_logger.warning(
                "Couldn't set mavros stream rate. Check if mavros is actually running and connected to the drone."
            )
        self.stream_rate_client.call_async(
            StreamRate.Request(stream_id=0, message_rate=30, on_off=True)
        )

    def _local_listener_callback(self, msg: PoseStamped):
        """
        Callback function that runs each time a new PoseStamped message is received.
        Adds the new pose and timestamp to the buffer and logs to file if log_dir is specified.
        """
        # Record the timestamp and convert message to a Pose object
        pose = Pose.from_mavros(msg)
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        curr_time = time.time()

        if self._testing_time_offset_sec is None and curr_time - timestamp > 1:
            self.console_logger.warning(
                f"Timestamp is more than a second behind current time: {curr_time - timestamp}, adding offset"
            )
            self._offset = curr_time - timestamp
        if self._testing_time_offset_sec is not None:
            timestamp += self._testing_time_offset_sec

        # Append the new pose to the buffer, keeping only the last `buffer_size` entries
        self.pose_buffer.append(pose)
        self.timestamp_buffer.append(timestamp)

        assert len(self.pose_buffer) == len(self.timestamp_buffer)
        if len(self.pose_buffer) > self.buffer_size:
            self.pose_buffer.pop(0)
            self.timestamp_buffer.pop(0)

        # If a log directory is specified, save the pose as JSON
        if self.data_log_dir is not None:
            if not self.data_log_dir.exists():
                self.data_log_dir.mkdir(parents=True)
            with open(
                self.data_log_dir
                / f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec}.json",
                "w",
            ) as f:
                json.dump(pose.to_dict(), f)

    def _global_listener_callback(self, msg: NavSatFix):
        self.latest_gps_msg = GlobalPosition(msg.latitude, msg.longitude, msg.altitude)

    def spin_thread(self):
        """Keeps the node spinning in a separate thread."""
        self.custom_executor.spin()

    def disconnect(self) -> None:
        """Safely shuts down the node."""
        self.custom_executor.shutdown(0)
        self.thread.join()
        self.destroy_node()

    def get_pose_at_time(self, time_sec: float, rosbag_testing=False) -> Pose:
        """
        Returns the pose at a specified timestamp by interpolating/extrapolating
        based on the buffered poses. Logs a warning if extrapolating more than a second out.

        Parameters:
            time (float): Target time in seconds since epoch.
            testing (bool): When testing with rosbag or other logs with old timestamps, enable this manually
        """

        if not self.pose_buffer:
            raise BufferError("Pose buffer is empty")

        if self._testing_time_offset_sec is None:
            self._testing_time_offset_sec = time_sec - self.timestamp_buffer[-1]
        time_sec -= 0 if rosbag_testing else self._testing_time_offset_sec

        # Case 1: Extrapolate if requested time is before the earliest timestamp
        if time_sec <= self.timestamp_buffer[0]:
            t0, pose0 = self.timestamp_buffer[0], self.pose_buffer[0]
            if len(self.pose_buffer) >= 2:
                t1, pose1 = self.timestamp_buffer[1], self.pose_buffer[1]
            else:
                self.console_logger.warning(
                    "Not enough data to extrapolate, returning last known pose"
                )
                return pose0  # Only one pose available
            if (self.timestamp_buffer[0] - time_sec) > 1.0:
                self.console_logger.warning(
                    "Extrapolating more than 1 second into the past"
                )
            return pose0.interpolate(pose1, (time_sec - t0) / (t1 - t0))

        # Case 2: Extrapolate if requested time is after the latest timestamp
        elif time_sec >= self.timestamp_buffer[-1]:
            if len(self.pose_buffer) >= 2:
                t0, pose0 = self.timestamp_buffer[-2], self.pose_buffer[-2]
                t1, pose1 = self.timestamp_buffer[-1], self.pose_buffer[-1]
            else:
                self.console_logger.warning(
                    "Not enough data to extrapolate, returning last known pose"
                )
                return self.pose_buffer[
                    -1
                ]  # just return the last pose cuz there's only one pose in the buffer

            if (time_sec - self.timestamp_buffer[-1]) > 1.0:
                self.console_logger.warning(
                    "Extrapolating more than 1 second into the future"
                )
            return pose0.interpolate(pose1, (time_sec - t0) / (t1 - t0))

        # Case 3: Interpolate between two poses within the buffer range
        else:
            if len(self.pose_buffer) < 2:
                raise BufferError(
                    "Pose buffer has insufficient values for interpolation"
                )
            idx = bisect.bisect_left(self.timestamp_buffer, time_sec)
            t0, pose0 = self.timestamp_buffer[idx - 1], self.pose_buffer[idx - 1]
            t1, pose1 = self.timestamp_buffer[idx], self.pose_buffer[idx]
            return pose0.interpolate(pose1, (time_sec - t0) / (t1 - t0))

    def get_global_pose(self) -> GlobalPosition:
        if self.latest_gps_msg is None:
            raise BufferError
        return GlobalPosition(
            self.latest_gps_msg.latitude,
            self.latest_gps_msg.longitude,
            self.latest_gps_msg.altitude,
        )

    def get_local_pose(self) -> Pose:
        """Alias for get_pose_at_time with hardcoded time argument to be current time."""
        return self.get_pose_at_time(time.time())

    def wait_until_ready(self, node):
        """Will finish once first local and global pose have been received. Node parameter for logging."""
        while True:
            try:
                pose = self.get_pose_at_time(time.time())
                self.console_logger.info(
                    f"Got initial pose: {pose}. Trying to get global pose..."
                )
                global_pos = self.get_global_pose()
                self.console_logger.info(f"Got global pose: {global_pos}")
                break
            except BufferError:
                self.console_logger.info("Waiting for pose provider to get pose...")
                time.sleep(1)
        self.console_logger.info("Pose provider ready.")
