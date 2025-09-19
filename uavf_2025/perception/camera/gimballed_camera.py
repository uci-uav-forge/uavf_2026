from pathlib import Path


from .camera import Camera, ImageMetadata
from typing import Type
from shared.types import Pose
import numpy as np
from scipy.spatial.transform import Rotation
from time import time
from abc import abstractmethod


class GimballedCamera(Camera):
    EULER_ORDER = (
        "YXZ"  # pitch, roll, yaw (right to left how the joints are physically attached)
    )

    def __init__(self, log_dir: str | Path | None = None):
        """
        Currently starts recording and logging as soon as constructed.
        This should be changed to after takeoff.
        """
        super().__init__(log_dir)

    @abstractmethod
    def set_absolute_position(self, yaw: float, pitch: float):
        pass

    @abstractmethod
    def set_gimbal_speed(self, yaw_speed: int, pitch_speed: int):
        pass

    @abstractmethod
    def do_autofocus(self):
        pass

    @abstractmethod
    def set_absolute_zoom(self, zoom_level: float):
        pass

    @abstractmethod
    def get_attitude(self) -> tuple[float, float, float]:
        """Returns current (yaw, pitch, roll)"""
        pass

    @abstractmethod
    def get_attitude_speed(self) -> tuple[float, float, float]:
        """Returns (yaw_speed, pitch_speed, roll_speed)"""
        pass

    @abstractmethod
    def get_zoom_level(self) -> float:
        pass

    def point_center(self):
        """
        Points to (0,0)
        """
        return self.set_absolute_position(0, 0)

    def point_down(self):
        """
        Points to (0, -90)
        """
        return self.set_absolute_position(0, -90)

    def get_attitude_interpolated(self, timestamp: float, offset: float = 1):
        """
        Gets the camera attitude at the point in time `timestamp - offset`
        """
        return self.metadata_buffer.get_interpolated(timestamp - offset).relative_pose

    def get_metadata(self) -> ImageMetadata:
        attitude = self.get_attitude()
        timestamp = time()

        return ImageMetadata(
            timestamp=timestamp,
            relative_pose=Pose(
                np.array(
                    [0, 0, -0.05]
                ),  # camera is roughly 5 cm under the flight controller
                Rotation.from_euler(
                    "YXZ", [-attitude[1], attitude[2], attitude[0]], degrees=True
                ),
                # rotation explanation:
                # attitude is yaw, pitch, roll,
                # but the gimbal is mounted such that the joints are yaw,
                # roll, and then pitch. Yaw is rotation around the z axis,
                # pitch is rotation around the y axis, and roll is rotation around x
                # The string "ZXY" specifies the order of the rotation axes, and then
                # we need to give it the angles in the order that the rotations are
                # applied.
            ),
            focal_len_px=self.get_focal_length_px(),
        )


class FixedGimballedCamera(GimballedCamera):
    def __init__(
        self, underlying_camera_class: Type[Camera], log_dir: str | Path | None = None
    ):
        super().__init__(log_dir)
        self._camera = underlying_camera_class()

    def take_image(self):
        return self._camera.take_image()

    def get_metadata(self):
        return self._camera.get_metadata()

    def get_focal_length_px(self) -> float:
        return self._camera.get_focal_length_px()

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
        return (0, -90, 0)

    def get_attitude_speed(self) -> tuple[float, float, float]:
        """Returns (yaw_speed, pitch_speed, roll_speed)"""
        return (0, 0, 0)

    def get_zoom_level(self) -> float:
        return 1
