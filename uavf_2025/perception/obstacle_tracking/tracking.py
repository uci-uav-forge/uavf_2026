import numpy as np
from perception.camera import Camera
from perception.camera.camera import ImageMetadata
from perception.obstacle_tracking.detection import ObstacleDetector
from perception.types import CameraPose, Image
from shared.types import Pose
import cv2

from .particle_filtering import ParticleFilter
from .types import Obstacle
from line_profiler import profile


class ObstacleTracker:
    """
    Composes inputs from the cameras and processes them through the detector and the localizer.

    NOTE: The delimiatations between this class and ParticleFilter are not that clear because I'm
    not sure about what are the limits of the algorithm itself vs. post-processing.
    """

    def __init__(
        self,
        cameras: list[Camera],
        obstacle_detector: ObstacleDetector,
        localizer: ParticleFilter,
    ):
        self._cameras = cameras
        self._obstacle_detector = obstacle_detector
        self._localizer = localizer

        self._last_timestamp: float | None = None

        self._debug_img: np.ndarray | None = None

    @profile
    def update(self, drone_pose: Pose) -> None:
        """
        Iterates the localizer with the latest images, and detections.
        """
        images, poses, sizes, metadata = self.get_image_data(drone_pose)

        timestamp = float(np.mean([metadatum.timestamp for metadatum in metadata]))
        time_elapsed = (
            (timestamp - self._last_timestamp)
            if self._last_timestamp is not None
            else 0
        )
        self._last_timestamp = timestamp

        debug_img = images[0].get_array().copy() if len(images) > 0 else None

        detections = [
            self._obstacle_detector(image) for image in images if image is not None
        ]

        if len(detections) == 0:
            print("no detections?")
            return

        if debug_img is not None:
            for det in detections[0]:
                cv2.rectangle(
                    debug_img,
                    (int(det.x - det.width // 2), int(det.y - det.height // 2)),
                    (int(det.x + det.width // 2), int(det.y + det.height // 2)),
                    (0, 0, 255),
                    2,
                )
        self._debug_img = debug_img
        self._localizer.update(detections, poses, sizes, time_elapsed)

    def get_debug_img(self) -> np.ndarray | None:
        return self._debug_img

    def reset(self) -> None:
        self._localizer.reset()

    def get_estimates(self) -> list[Obstacle]:
        return self._localizer.get_estimates()

    def get_particles(self):
        return self._localizer.get_particles()

    def get_image_data(self, drone_pose: Pose):
        """
        Gets the images, poses, sizes, and metadata from the cameras.
        """
        images: list[Image] = []
        sizes: list[tuple[int, int]] = []

        metadata: list[ImageMetadata] = []
        poses: list[CameraPose] = []

        assert len(self._cameras) == 1

        for camera in self._cameras:
            image = (
                camera.get_latest_image() if camera.recording else camera.take_image()
            )
            metadatum = camera.get_metadata()
            if image is not None:
                images.append(image)
                sizes.append((image.height, image.width))

                poses.append(camera.get_world_pose(drone_pose))
                metadata.append(metadatum)

        assert len(images) <= 1, f"{len(images)} images > {len(self._cameras)} cameras"

        return images, poses, sizes, metadata
