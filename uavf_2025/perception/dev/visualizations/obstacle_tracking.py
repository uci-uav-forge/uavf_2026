from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
from perception.camera.replay_camera import ReplayCamera
from perception.lib.yolo import TiledYolo
import torch
from tqdm import tqdm
from matplotlib import widgets, pyplot as plt

from perception.camera import DummyCamera, ImageMetadata
from perception.obstacle_tracking import (
    ObstacleTracker,
    ParticleFilter,
    DummyObstacleDetector,
)
from perception.types import Bbox2D, Image
from shared.types import Pose


class ObstacleTrackerVisualizer:
    def __init__(
        self,
        tracker: ObstacleTracker,
        drone_poses: Sequence[Pose],
        iterations: int = 300,
        frames: list[np.ndarray] | None = None,
        run_simulation: bool = True,
        view: bool = True,
    ):
        """
        Parameters
        ----------
        tracker : ObstacleTracker
            The obstacle tracker to visualize.
        drone_poses : Sequence[Pose]
            The poses of the drone at each timestep. If there are fewer poses
            than iterations, the visualization will stop when the poses run out.
        iterations : int
            Maximum number of iterations to run the simulation for.
        frames : list[np.ndarray] | None
            Camera frames to display alongside the 3D visualization.
        run_simulation : bool
            If True, immediately run the simulation and store the states.
            Otherwise, the simulation can be run with `run_simulation()`.
        view : bool
            If True, displays the visualization immediately.
            Otherwise, the visualization can be displayed with `visualize()`.
            Will only take effect if run_simulation is True.
        """
        self._tracker = tracker
        self._drone_poses = drone_poses
        self._iterations = iterations
        self._frames = frames

        self._states = self._simulate() if run_simulation else None

        if run_simulation and view:
            self.visualize()

    def run_simulation(self) -> None:
        """
        Runs the simulation and stores the states.
        """
        self._states = self._simulate()

    def visualize(self) -> None:
        """
        Displays the visualization.

        Reference: https://stackoverflow.com/a/68702873
        """
        if self._states is None:
            raise RuntimeError("Simulation must be run before visualization")

        fig = plt.figure()

        ax1 = fig.add_axes((0, 0, 1, 0.8), projection="3d")  # 3D plot axis

        ax2 = fig.add_axes((0.1, 0.85, 0.8, 0.1))  # slider axis
        slider = widgets.Slider(ax2, "Time Step", 0, len(self._states) - 1, valinit=0)

        if self._frames is not None:
            assert len(self._frames) == self._iterations

            ax3 = fig.add_axes((0.1, 0.7, 0.8, 0.15))  # camera frame axis
            ax3.set_title("Camera Frame")

        def update(iteration: float):
            if self._states is None:
                raise RuntimeError("Simulation must be run before visualization")

            iteration = round(iteration)
            if iteration < 0:
                iteration = 0

            ax1.clear()
            ax1.set_title(f"Iteration {iteration}")

            particles = self._states[iteration]
            ax1.scatter(particles[:, 0], particles[:, 1], particles[:, 2])

            drone_pose = self._drone_poses[iteration]
            direction = drone_pose.rotation.apply([0, 0, 1])

            arrow_length = torch.mean(torch.tensor(particles).norm(dim=1)).item() / 10
            ax1.quiver(*drone_pose.position, *direction, length=arrow_length, color="r")

            if self._frames is not None:
                assert ax3, "Axis should have been created"  # type: ignore
                ax3.clear()
                ax3.imshow(self._frames[iteration])
                ax3.axis("off")

        slider.on_changed(update)
        update(0)

        plt.show()

    def _simulate(self) -> list[np.ndarray]:
        states: list[np.ndarray] = []

        for _, drone_pose in tqdm(
            zip(range(self._iterations), self._drone_poses),
            total=self._iterations,
            desc="Simulating particle states",
        ):
            self._tracker.update(drone_pose)
            states.append(self._tracker.get_particles().cpu().numpy())

        return states


def static_tracking(iterations: int = 100):
    """
    Smoke test where neither the camera nor the obstacle detection move.
    """
    camera = DummyCamera(
        [Image(np.zeros((1080, 1920, 3), dtype=np.uint8)) for _ in range(iterations)],
        [ImageMetadata(0, Pose.identity(), 1000) for _ in range(iterations)],
    )

    # Dummy obstacle detector with one obstacle at the center of the image
    detector = DummyObstacleDetector([[Bbox2D(0.5, 0.5, 0.25, 0.25)]])

    localizer = ParticleFilter(num_particles=2000, device=torch.device("cpu"))

    tracker = ObstacleTracker([camera], detector, localizer)

    drone_poses = repeat(Pose.identity(), iterations)

    ObstacleTrackerVisualizer(tracker, list(drone_poses), iterations)


def video_tracking(source_dir: Path, iterations: int = 100):
    camera = ReplayCamera(source_dir)

    detector = TiledYolo.from_path(
        Path("uavf_2025/perception/lib/coco_yolo11n.pt"), num_classes=80
    )

    localizer = ParticleFilter(num_particles=2000, device=torch.device("cpu"))

    tracker = ObstacleTracker(
        [camera],
        lambda image: list(result.bbox for result in detector.detect(image, 0)),
        localizer,
    )

    drone_poses = repeat(Pose.identity(), iterations)

    ObstacleTrackerVisualizer(tracker, list(drone_poses), iterations)


def dummy_video_tracking(height: int = 1080, width: int = 1920):
    """
    Generates a video and corresponding bounding boxes for a square moving from
    left to right while getting smaller (i.e., equivalent to moving diagonally away),
    then runs the tracking algorithm and visualization on it.
    """
    num_frames = 100

    centers = np.linspace(0.25, 0.75, num_frames)
    sizes = np.linspace(0.25, 0.05, num_frames)

    images: list[Image[np.ndarray]] = []
    bboxes: list[list[Bbox2D]] = []

    for center, size in zip(centers, sizes):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(
            img,
            (int((center - size / 2) * width), int((center - size / 2) * height)),
            (int((center + size / 2) * width), int((center + size / 2) * height)),
            (255, 255, 255),
            -1,
        )

        images.append(Image(img))
        bboxes.append([Bbox2D(center, center, size, size)])

    camera = DummyCamera(
        images, [_dummy_metadata_factory("") for _ in range(len(images))]
    )

    detector = DummyObstacleDetector(bboxes)

    localizer = ParticleFilter(num_particles=2000, device=torch.device("cpu"))

    tracker = ObstacleTracker([camera], detector, localizer)

    drone_poses = list(repeat(Pose.identity(), num_frames))
    ObstacleTrackerVisualizer(tracker, drone_poses, num_frames)


def _dummy_metadata_factory(name: str) -> ImageMetadata:
    return ImageMetadata(int(name) if name.isnumeric() else 0, Pose.identity(), 1000)


def generate_dummy_metadata(
    directory: Path,
    factory: Callable[[str], ImageMetadata] = _dummy_metadata_factory,
    overwrite=False,
):
    """
    Generates metadata files for a directory of images.
    """
    executor = ThreadPoolExecutor(4)

    def make_and_save_metadata(img_path: Path):
        metadata_path = img_path.with_suffix(".json")

        if not overwrite and metadata_path.exists():
            return

        metadata_path = metadata_path
        metadata = factory(img_path.name)
        metadata.save(metadata_path)

    for img_path in directory.glob("*.png"):
        executor.submit(make_and_save_metadata, img_path)


if __name__ == "__main__":
    static_tracking()
