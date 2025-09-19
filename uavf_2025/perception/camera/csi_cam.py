from .arducam_libraries.Focuser import Focuser
from .arducam_libraries.JetsonCamera import GstCamera
from perception.camera.camera import Camera, ImageMetadata
from shared.types import Pose
from enum import Enum
from scipy.spatial.transform import Rotation
import numpy as np
from pathlib import Path
import cv2 as cv
from perception.types import Image
import time


class CSICam(Camera):
    class ResolutionOption(Enum):
        R4K = (3840, 2160)
        R1080P = (1920, 1080)
        R720P = (1280, 720)
        R480P = (640, 480)

    def __init__(
        self,
        log_dir: str | Path | None = None,
        resolution: ResolutionOption = ResolutionOption.R4K,
        flipped=True,  # because of how they're mounted we might have to flip them sometimes.
    ):
        """
        TODO: make resolution choosable. Hard-coded in JetsonCamera rn.
        """
        super().__init__(log_dir)
        pipeline = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                resolution.value[0],
                resolution.value[1],
                30,
                0 if flipped else 2,
                resolution.value[0],
                resolution.value[1],
            )
        )

        self._camera = GstCamera(pipeline)
        self._focuser = Focuser(9)
        self.set_focus(450)
        self._relative_pose = Pose(
            np.array([0.1, 0, -0.05]), Rotation.from_euler("y", 90, degrees=True)
        )
        self._resolution = resolution
        self._flipped = flipped

    def take_image(self) -> Image:
        frame = self._camera.getFrame()
        if frame is None:
            return None
        if self._flipped:
            frame = cv.rotate(frame, cv.ROTATE_180)
        return Image(frame)

    def get_metadata(self) -> ImageMetadata:
        return ImageMetadata(
            timestamp=time.time(),
            relative_pose=self._relative_pose,
            focal_len_px=self.get_focal_length_px(),
        )

    def get_focal_length_px(self):
        return 2516 * self._resolution.value[0] / 4056
        # Using arducam specs. 3.9mm focal length, 1.55 micrometer pixel size, 4056 pixels horizontal
        # We should actually do camera calibration at some point

    def set_focus(self, value: int):
        """
        value needs to be between 0 and 1000
        """
        assert 0 <= value <= 1000, f"{value} out of range [0,1000]"
        self._focuser.set(Focuser.OPT_FOCUS, value)
