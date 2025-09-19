from .siyi_cam import SiyiCam
from pathlib import Path
import warnings


class A8Camera(SiyiCam):
    def __init__(
        self,
        log_dir: str | Path | None = None,
    ):
        super().__init__(log_dir, cam_ip="192.168.144.26")

    def get_focal_length_px(self):
        return 1219  # hard-coded value for a8 mini

    def set_absolute_zoom(self, zoom_level: float):
        warnings.warn(
            "warning: trying to send zoom command to a8 mini which it doesnt work on yet"
        )

    def get_zoom_level(self) -> float:
        return 1.0
