import rclpy
import rclpy.node
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import threading
from rclpy.executors import SingleThreadedExecutor

from shared import ensure_ros
from perception.camera import Camera, ImageMetadata
from perception.types import Image
from shared.types import Pose
from time import time
from pathlib import Path
import subprocess
import json


class GazeboCamera(Camera, rclpy.node.Node):
    @ensure_ros
    def __init__(
        self,
        log_dir: str | Path | None,
        img_topic="/camera/image",
        relative_pose: Pose = Pose.identity(),
    ):
        Camera.__init__(self, log_dir)
        rclpy.node.Node.__init__(self, "uavf_2025_gazebo_camera_stream")
        self._topic_name = img_topic
        MAX_FPS = 10  # TODO: experiment with increasing this
        self.subscription = self.create_subscription(
            ROSImage, img_topic, self.listener_callback, MAX_FPS
        )
        self.bridge = CvBridge()
        self.latest_msg: ROSImage | None = None
        self.custom_executor = SingleThreadedExecutor()
        self.custom_executor.add_node(self)
        self.bad_count = 0
        self.last_frame_time = time()
        self._relative_pose = relative_pose
        self._focal_len_px = None
        thread = threading.Thread(target=self.spin_thread, daemon=True)
        thread.start()

    def listener_callback(self, msg: ROSImage):
        self.latest_msg = msg

    def spin_thread(self):
        self.custom_executor.spin()

    def take_image(self):
        if self.latest_msg is None:
            self.bad_count += 1
            return None
        self.bad_count = 0
        cv_img = self.bridge.imgmsg_to_cv2(self.latest_msg, desired_encoding="bgr8")
        self.latest_msg = None
        return Image(cv_img)

    def get_metadata(self) -> ImageMetadata:
        return ImageMetadata(
            timestamp=time(),
            relative_pose=self._relative_pose,
            focal_len_px=self.get_focal_length_px(),
        )

    def get_focal_length_px(self) -> float:
        if self._focal_len_px is None:
            command = [
                "gz",
                "topic",
                "-e",
                "--json-output",
                "-t",
                "/world/map/model/iris/link/tilt_link/sensor/camera/camera_info"
                if self._topic_name == "/camera/image"
                else self._topic_name.replace("image", "camera_info"),
                "-n",
                "1",
            ]

            # Execute the command and capture the output
            result = subprocess.run(command, capture_output=True, text=True)

            # Check if the command was successful
            if result.returncode != 0:
                raise RuntimeError(
                    f"Command {' '.join(command)} failed with error: {result.stderr}"
                )

            # Parse the JSON output into a Python dictionary
            try:
                data_dict = json.loads(result.stdout)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to decode JSON: {e}")
            self._focal_len_px = data_dict["intrinsics"]["k"][0]

        return self._focal_len_px
