import subprocess
from .gazebo import GazeboGimballedCamera
from .siyi import A8Camera, ZR10Camera
from .gimballed_camera import GimballedCamera, FixedGimballedCamera
from .mock_camera import MockGimballedCamera
from .usb_cam import USBCam
import os


def _check_topic_exists(topic_name):
    """Checks if a ROS2 topic exists using the `ros2 topic info` command.

    Args:
      topic_name: The name of the topic to check.

    Returns:
      True if the topic exists, False otherwise.
    """
    result = subprocess.run(
        ["ros2", "topic", "info", topic_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.returncode == 0


def make_gimballed_camera(log_dir) -> GimballedCamera:
    if os.environ.get("MOCK_CAMERA", False):
        return MockGimballedCamera(log_dir)
    if _check_topic_exists("/camera/image"):
        return GazeboGimballedCamera(log_dir)
    else:
        try:
            return FixedGimballedCamera(USBCam, log_dir)
        except RuntimeError:
            try:
                return A8Camera(log_dir)
            except RuntimeError:
                return ZR10Camera(log_dir)
