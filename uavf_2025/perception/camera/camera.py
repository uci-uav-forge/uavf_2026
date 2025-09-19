import json
import threading
import time
import traceback
import warnings
from abc import ABC, abstractmethod
from bisect import bisect_left
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation

from perception.types import CameraPose, Image
from shared.types import Pose


class ImageBuffer:
    """
    Buffer for one Image implementing a Lock for multithreading.
    """

    def __init__(self):
        self.image = None
        self.lock = threading.Lock()

    def put(self, image: Image[np.ndarray]):
        with self.lock:
            self.image = image

    def get_latest(self) -> Image[np.ndarray] | None:
        with self.lock:
            return self.image


class ImageMetadata(NamedTuple):
    timestamp: float
    relative_pose: Pose
    focal_len_px: float

    def save(self, path: Path):
        json.dump(self._asdict(), open(path, "w"))

    @staticmethod
    def load(path: Path) -> "ImageMetadata":
        return ImageMetadata(**json.load(open(path)))


class InterpolatableBuffer:
    def __init__(self):
        self._queue: deque[ImageMetadata] = deque(maxlen=128)
        self.lock = threading.Lock()

    def push(self, datum: ImageMetadata):
        with self.lock:
            self._queue.append(datum)

    def get_latest(self) -> ImageMetadata | None:
        with self.lock:
            return self._queue[-1] if self._queue else None

    def get_interpolated(self, timestamp: float) -> ImageMetadata:
        with self.lock:
            if self._queue[-1].timestamp < timestamp:
                return self._queue[-1]
            if self._queue[0].timestamp > timestamp:
                return self._queue[0]

            idx = bisect_left([d.timestamp for d in self._queue], timestamp)

            before = self._queue[idx - 1]
            after = self._queue[idx]
            proportion = (timestamp - before.timestamp) / (
                after.timestamp - before.timestamp
            )
            new_pose = before.relative_pose.interpolate(after.relative_pose, proportion)
            new_f = before.focal_len_px + proportion * (
                after.focal_len_px - before.focal_len_px
            )

            return ImageMetadata(
                timestamp=timestamp,
                relative_pose=new_pose,
                focal_len_px=new_f,
            )


class CameraLogger:
    """
    Class encapsulating multithreaded logging of images and metadata.

    The thread pool will be automatically destroyed when the CameraLogger object is destroyed.
    """

    def __init__(self, log_dir: str | Path, max_threads: int = 2):
        self.log_dir = Path(log_dir)
        self._prep_log_dir(self.log_dir)

        self.datum_id = 0
        self.pool = ThreadPoolExecutor(max_workers=max_threads)

    @staticmethod
    def _prep_log_dir(log_dir: Path):
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)

    def log_async(self, image: Image, metadata: ImageMetadata):
        """
        Asynchronously logs the image and metadata to the log directory.
        """
        self.pool.submit(self._log_to_file, image, metadata)

    def _log_to_file(self, image: Image, metadata: ImageMetadata):
        try:  # since this runs in a pool executor it won't log exceptions unless we do this
            image.save(self.log_dir / f"{self.datum_id}.jpg")

            json.dump(
                {
                    "timestamp": metadata.timestamp,
                    "focal_len_px": metadata.focal_len_px,
                    "relative_pose": metadata.relative_pose.to_dict(),
                },
                open(self.log_dir / f"{self.datum_id}.json", "w"),
            )
            self.datum_id += 1
        except Exception:
            traceback.print_exc()


class Camera(ABC):
    def __init__(self, log_dir: str | Path | None = None):
        """
        Paramters
        ----------
        relative_pose : Pose
            The pose of the camera relative to the drone.
        """
        self._log_dir = log_dir

        self._pose_id = 0

        # Buffer for taking the latest image, logging it, and returning it in get_latest_image
        self.buffer = ImageBuffer()

        self.recording_thread: threading.Thread | None = None
        self.recording = False

        # Controls whether images and data are submitted to the `threaded_logger`
        if log_dir:
            self.threaded_logger = CameraLogger(Path(log_dir))
            self.logging = True
        else:
            self.logging = False

        self.metadata_buffer = InterpolatableBuffer()

    @abstractmethod
    def take_image(self) -> Image[np.ndarray] | None:
        pass

    @abstractmethod
    def get_metadata(self) -> ImageMetadata:
        pass

    @abstractmethod
    def get_focal_length_px(self) -> float:
        return NotImplemented

    def set_log_dir(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)

    def _recording_worker(self):
        """
        Worker function that continuously gets frames from the stream and puts them in the buffer as well as logging them.
        """
        while self.recording:
            try:
                image = self.take_image()
                if image is None:
                    print("Failed to get image")
                    time.sleep(0.1)
                    continue

            except Exception:
                # Waits 100ms before trying again
                print("Error getting frame")
                traceback.print_exc()
                time.sleep(0.1)
                continue

            try:
                metadata = self.get_metadata()

            except Exception:
                # Waits 100ms before trying again
                print("Error getting metadata")
                traceback.print_exc()
                time.sleep(0.1)
                continue

            self.buffer.put(image)

            # For interpolation
            self.metadata_buffer.push(metadata)

            if self.logging:
                self.threaded_logger.log_async(image, metadata)

            time.sleep(0.1)

    def start_recording(self):
        if self.recording or not self.logging:
            return
        self.recording_thread = threading.Thread(target=self._recording_worker)
        self.recording = True
        self.recording_thread.start()
        self.start_logging()

    def start_logging(self):
        self.logging = True

    def stop_logging(self):
        self.logging = False

    def stop_recording(self):
        if self.recording_thread:
            self.stop_logging()
            self.recording = False
            self.recording_thread.join()
            self.recording_thread = None

    def get_latest_image(self) -> Image[np.ndarray] | None:
        """
        Returns the latest Image (HWC) from the buffer.
        """
        if not self.recording:
            warnings.warn(
                "Trying to get frame from buffer while camera is not recording."
            )
        return self.buffer.get_latest()

    @staticmethod
    def combine_drone_rot(cam_rot: Rotation, drone_rot: Rotation) -> Rotation:
        """
        Composes the drone pose with the relative pose of the camera to the drone.
        This method is a simple one-liner that exists so we don't mess up the order of the operation,
        because mutliplying rotations is not commutative. This method is also overridden for some camera types
        where the cam rotation is independent of some axes of the drone's rotation.
        """
        return drone_rot * cam_rot

    def get_world_pose(self, drone_pose: Pose) -> CameraPose:
        """
        Composes the drone pose with the relative pose of the camera to the drone.
        """
        position = drone_pose.position + drone_pose.rotation.apply(
            self.get_metadata().relative_pose.position
        )
        rotation = Camera.combine_drone_rot(
            self.get_metadata().relative_pose.rotation, drone_pose.rotation
        )

        ret = CameraPose(position, rotation, self.get_focal_length_px())

        self._pose_id += 1

        return ret
