from abc import abstractmethod
from pathlib import Path
from typing import Sequence, Protocol

from perception.types import Bbox2D, Image
from ultralytics import YOLO


CURRENT_DIR = Path(__file__).absolute().parent


class ObstacleDetector(Protocol):
    @abstractmethod
    def __call__(self, image: Image) -> list[Bbox2D]: ...


class YoloObstacleDetector(ObstacleDetector):
    def __init__(self):
        self.yolo = YOLO(CURRENT_DIR / "drones-11n-v2.pt")

    def __call__(self, image: Image) -> list[Bbox2D]:
        return [
            Bbox2D(*result.boxes.xywh[0].cpu().numpy())
            for result in self.yolo.predict(
                image.get_array(), verbose=False, imgsz=1280, conf=0.1
            )[0]
        ]


class DummyObstacleDetector(ObstacleDetector):
    """
    Loops through the detection results provided in the constructor.
    """

    def __init__(self, bboxes: Sequence[Sequence[Bbox2D]]):
        self._detections_generator = self._make_detections_loop(bboxes)

    def __call__(self, image: Image) -> list[Bbox2D]:
        return next(self._detections_generator)

    @staticmethod
    def _make_detections_loop(bbox_batches: Sequence[Sequence[Bbox2D]]):
        while True:
            for batch in bbox_batches:
                yield list(batch)
