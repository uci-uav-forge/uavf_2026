from .siyi_cam import SiyiCam
from pathlib import Path
import warnings


class ZR10Camera(SiyiCam):
    def __init__(
        self,
        log_dir: str | Path | None = None,
    ):
        super().__init__(log_dir)

    def get_focal_length_px(self):
        zoom_level = self.get_zoom_level()
        if 1 <= zoom_level <= 10:
            warnings.warn(f"ZR10 zoom is set to nonsensical value: {zoom_level}")
            return 0
        else:
            return (
                90.9 + 1597.2 * zoom_level
            )  # hard-coded based on doing calibrations and linear regression
