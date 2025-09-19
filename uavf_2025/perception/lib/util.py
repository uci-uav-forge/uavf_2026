import logging
from pathlib import Path

import psutil
import numpy as np
from scipy.spatial.transform import Rotation
from shared.types import Pose


# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
class ColoredFormatter(logging.Formatter):
    def __init__(self, template: str):
        super().__init__(template)

        grey = "\x1b[38;20m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"

        self.FORMATS = {
            logging.DEBUG: grey + template + reset,
            logging.INFO: grey + template + reset,
            logging.WARNING: yellow + template + reset,
            logging.ERROR: red + template + reset,
            logging.CRITICAL: bold_red + template + reset,
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def create_console_logger(
    logs_path: Path, name: str, capture_all_logs: bool = True
) -> logging.Logger:
    logger: logging.Logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set the overall logging level

    # Create file handler
    file_handler = logging.FileHandler(logs_path / f"{name}.log")
    if capture_all_logs:
        file_handler.setLevel(logging.DEBUG)  # Capture all logs

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Only show INFO and above on console

    # Define log format
    file_formatter = logging.Formatter("%(asctime)s %(message)s")
    console_formatter = ColoredFormatter(
        "%(name)s: %(asctime)s %(message)s"
    )  # Prepend logger name

    # Apply formatters
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


class ProcessMemLogger:
    def __init__(self):
        # important: it gets the current process wherever this class is initialized
        self.proc = psutil.Process()

    def __call__(self) -> str:
        vm = psutil.virtual_memory()
        mem_used = vm.total - vm.available
        return f"{self.proc.memory_info().rss} {mem_used}"


CV_WORLD_FRAME_TO_WORLD_FRAME_MAT_ROTATION = np.array(
    [[0, 0, 1], [-1, 0, 0], [0, 1, 0]], dtype=np.float32
)

FLIP_Z_MATRIX = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -1],
    ]
)
CV_WORLD_FRAME_TO_WORLD_FRAME_MAT = (
    FLIP_Z_MATRIX @ CV_WORLD_FRAME_TO_WORLD_FRAME_MAT_ROTATION
)

CV_TO_WORLD_ROT = Rotation.from_matrix(CV_WORLD_FRAME_TO_WORLD_FRAME_MAT)


def cv_to_pose(rvec, tvec):
    """
    opencv camera rvec and tvec to drone pose in ardupilot frame, taking in to account hard-coded camera angle
    """
    r_cv = Rotation.from_rotvec(rvec)
    t_cv = tvec

    def inv(x):
        return x.inv()

    pose_r = inv(CV_TO_WORLD_ROT * r_cv * inv(CV_TO_WORLD_ROT))
    pose_t = -CV_WORLD_FRAME_TO_WORLD_FRAME_MAT @ r_cv.inv().as_matrix() @ t_cv
    new_pose = Pose(
        pose_t,
        pose_r,
    )
    return new_pose


A = CV_TO_WORLD_ROT.inv()


def pose_to_cv(pose: Pose):
    """
    drone pose in ardupilot frame to opencv camera rvec and tvec, taking into account hard-coded camera angle
    """

    rvec_rot = A * pose.rotation.inv() * CV_TO_WORLD_ROT
    tvec = -rvec_rot.apply(pose.position @ CV_WORLD_FRAME_TO_WORLD_FRAME_MAT)
    return rvec_rot.as_rotvec(), tvec
