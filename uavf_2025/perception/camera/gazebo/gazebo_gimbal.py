import subprocess
import threading

from mavros_msgs.msg import OverrideRCIn
from sensor_msgs.msg import JointState
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, HistoryPolicy

import numpy as np

from shared import ensure_ros
from perception.camera.gimballed_camera import GimballedCamera
from .gazebo_cam import GazeboCamera
from pathlib import Path


class GazeboGimballedCamera(GimballedCamera, GazeboCamera):
    @ensure_ros
    def __init__(self, log_dir: str | Path | None = None):
        GazeboCamera.__init__(self, log_dir)
        self.attitude_subscriber = self.create_subscription(
            JointState,
            "/joint_states",
            self._attitude_callback,
            QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST),
        )

        self.rc_override_publisher = self.create_publisher(
            OverrideRCIn,
            "/mavros/rc/override",
            QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST),
        )

        self.custom_executor = SingleThreadedExecutor()
        self.custom_executor.add_node(self)

        self.zoom_level = 1  # default zoom level

        self.rc_overrides = np.zeros(18, dtype=np.uint16)

        # defaults for gimbal pointing straight ahead
        self.rc_overrides[8] = 1500  # yaw
        self.rc_overrides[9] = 1100 + int(90 / 115 * 800)  # pitch

        self.message_pub_timer = self.create_timer(0.1, self.publish_messages)
        self.last_attitude = None

        self.ros_spin_thread = threading.Thread(
            target=self.custom_executor.spin, daemon=True
        )
        self.ros_spin_thread.start()

    def publish_messages(self):
        self.rc_override_publisher.publish(OverrideRCIn(channels=self.rc_overrides))

    def do_autofocus(self):
        pass  # gazebo camera can't be out of focus because they didn't implement depth of field :(

    def set_absolute_zoom(self, zoom_level):
        command = f'gz topic -t "/model/iris/sensor/camera/zoom/cmd_zoom" -m gz.msgs.Double -p "data: {zoom_level:.01f}"'
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        self.zoom_level = zoom_level

    def set_absolute_position(self, yaw: float, pitch: float) -> None:
        """
        Params
        --
         - yaw [int] -135~135
         - pitch [int] -90~25
        both are in degrees
        """
        # RC channels 9 and 10, but zero-indexed
        # RC inputs are between 1100 and 1900 for min and max rotation values
        self.rc_overrides[8] = 1100 + int((yaw + 135) / 270 * 800)
        pitch_range = 25 - (-90)
        self.rc_overrides[9] = 1100 + int((pitch + 90) / pitch_range * 800)

    def set_gimbal_speed(self, yaw_speed: int, pitch_speed: int):
        raise NotImplementedError()

    def _attitude_callback(self, msg: JointState):
        names = msg.name
        angles = msg.position
        velocities = msg.velocity
        for name, angle, velocity in zip(names, angles, velocities):
            if name == "tilt_joint":
                pitch = -np.rad2deg(
                    angle
                )  # we want negative pitch to be down, and gazebo had it flipped by default
                pitch_speed = np.rad2deg(velocity)
            elif name == "yaw_joint":
                yaw = np.rad2deg(angle)
                yaw_speed = np.rad2deg(velocity)
            elif name == "roll_joint":
                roll = np.rad2deg(angle)
                roll_speed = np.rad2deg(velocity)
        self.last_attitude = (yaw, pitch, roll)
        self.last_attitude_speed = (yaw_speed, pitch_speed, roll_speed)

    def get_attitude(self) -> tuple[float, float, float]:
        if self.last_attitude is None:
            raise RuntimeError("No attitude data available. Is gazebo running?")
        return self.last_attitude

    def get_attitude_speed(self) -> tuple[float, float, float]:
        return self.last_attitude_speed

    def get_zoom_level(self) -> float:
        """
        This is a little bugged because the camera zoom in simulation
        doesn't instantly change when we command it, but this variable does.
        """
        return self.zoom_level
