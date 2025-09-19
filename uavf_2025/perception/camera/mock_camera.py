from numpy import ndarray
from perception.types import Image
from .gimballed_camera import GimballedCamera
from .camera import Camera, ImageMetadata
from time import time
from shared.types import Pose
import numpy as np


class MockCamera(Camera):
    def __init__(self, log_dir):
        super().__init__(None)

    def take_image(self) -> Image[ndarray] | None:
        return Image(np.random.randint(0, 255, size=(1080, 1920, 3)))

    def get_metadata(self) -> ImageMetadata:
        return ImageMetadata(time(), Pose.identity(), 1)

    def get_focal_length_px(self) -> float:
        return 1


class MockGimballedCamera(GimballedCamera, MockCamera):
    def __init__(self, log_dir):
        GimballedCamera.__init__(self, None)
        MockCamera.__init__(self, None)

    def set_absolute_position(self, yaw: float, pitch: float):
        pass

    def set_gimbal_speed(self, yaw_speed: int, pitch_speed: int):
        pass

    def do_autofocus(self):
        pass

    def set_absolute_zoom(self, zoom_level: float):
        pass

    def get_attitude(self) -> tuple[float, float, float]:
        """Returns current (yaw, pitch, roll)"""
        return (0, 0, 0)

    def get_attitude_speed(self) -> tuple[float, float, float]:
        """Returns (yaw_speed, pitch_speed, roll_speed)"""
        return (0, 0, 0)

    def get_zoom_level(self) -> float:
        return 1
