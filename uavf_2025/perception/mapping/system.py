import logging
from perception.types import Image, CameraPose
from .mapper import Mapper

import numpy as np
import cv2
from pathlib import Path
from time import strftime


class MapperSystem:
    """Mapper interface for perception.py"""

    LOG_MSG_PREFIX = "[Mapping/MapperSystem]: "

    def __init__(
        self,
        console_logger: logging.Logger,
        logging_path: Path,
        roi_coords_local: np.ndarray,
    ):
        """
        roi_coords_local needs to be the perimeter of the region of interest to be mapped,
        with shape (3, 2).

        For the purposes of getting an image back aligned to the ROI, the coordinates
        will be treated like (top_left, top_right, bottom_left)

        We assume flat ground with z=0. Units will be meters.
        """

        self._console_logger = console_logger
        self._mapper = Mapper(console_logger)

        self.is_active = False
        self.logging_path = logging_path
        self.index = 0

    def start_stitching(self) -> None:
        self.is_active = True

    def stop_stitching(self) -> None:
        self.is_active = False

    def stitch(self, image: Image, _camera_pose: CameraPose):
        """
        Add an image to the map using its camera pose
        """
        if self.is_active:
            self._console_logger.debug(self.LOG_MSG_PREFIX + f" Stitching {self.index}")
            success = self._mapper.stitch(
                cv2.cvtColor(image.get_array(), cv2.COLOR_RGB2RGBA)
            )
            if not success:
                self._console_logger.warning(
                    self.LOG_MSG_PREFIX + f" Stitching failed at {self.index}"
                )

            # self.save_map()  # DEBUG

            self.index += 1

    def clear(self) -> None:
        """
        Clears existing stored/aggregated map
        """
        self._mapper.clear()

        self.index = 0

    def get_map(self, _resolution: float) -> Image:
        """
        Get the map of the region of interest, with the given resolution.
        The resolution is the size of each pixel in meters.
        """
        return Image(np.array(self._mapper.get_raw_map()))

    def save_map(self) -> bool:
        timestamp = strftime("%Y-%m-%d_%H-%M")
        fname = str(self.logging_path / f"{timestamp}[{self.index}].png")

        self._console_logger.info(self.LOG_MSG_PREFIX + f" Saving map [{fname}]")
        if (map := self._mapper.get_raw_map()) is not None:
            success = cv2.imwrite(fname, map)
            if success:
                self._console_logger.debug(self.LOG_MSG_PREFIX + " Map saved")
                return True
            else:
                self._console_logger.error(self.LOG_MSG_PREFIX + " Map failed to save")
                return False
        else:
            self._console_logger.warning(
                self.LOG_MSG_PREFIX
                + " Map is None (attempting to save map when there are no map to save)"
            )
            return False

    def get_coverage_image(self):
        img = np.zeros((1000, 1000, 1), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (900, 900), 20, -1)
        cv2.rectangle(img, (400, 400), (600, 600), 00, -1)
        return img
