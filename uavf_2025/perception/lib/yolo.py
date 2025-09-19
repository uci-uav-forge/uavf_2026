from pathlib import Path

import numpy as np
from ultralytics import YOLO

from perception.types import Bbox2D, Image, TargetBBox
from line_profiler import profile


class TiledYolo:
    def __init__(
        self,
        model_path: Path,
        num_classes: int,
        tile_size: tuple[int, int] = (640, 640),
        overlap_ratios: tuple[float, float] = (0.2, 0.2),
    ):
        """
        This should not be consumed externally. Use one of the factory methods.
        """
        self._model = YOLO(model_path)
        self._num_classes = num_classes
        self._tile_size = tile_size
        self._overlap_ratios = overlap_ratios

        self._detection_counter = Counter()

    @staticmethod
    def from_path(
        model_path: Path,
        num_classes: int,
        tile_size: tuple[int, int] = (640, 640),
        overlap_ratios: tuple[float, float] = (0.2, 0.2),
        confidence_threshold: float = 0.5,
    ):
        """
        Create a TiledYolo object from a model file.

        Args:
            model_path (Path): Path to the model file.
            model_type (str, optional): Model type. Defaults to "yolov11".
            tile_size (tuple[int, int], optional): Size of the tiles (height, width).
            overlap_ratios (tuple[float, float], optional): Overlap ratios (height, width).
            confidence_threshold (float, optional): Confidence threshold. Defaults to 0.5.
        """

        return TiledYolo(model_path, num_classes, tile_size, overlap_ratios)

    @profile
    def detect(self, image: Image, img_id: int):
        """
        Detect objects in the given image.

        Args:
            image (Image): Image to detect objects in.

        Returns:
            list[dict]: List of detected objects.
        """
        tiles = image.generate_tiles(
            self._tile_size[0], int(self._overlap_ratios[0] * self._tile_size[0])
        )

        imgs_list = [t.img.get_array() for t in tiles]
        predictions_list = self._model.predict(imgs_list, verbose=False)

        return [
            TargetBBox(
                bbox=Bbox2D.from_xyxy(
                    pred.bbox.to_xyxy() + np.array([tile.x, tile.y, tile.x, tile.y])
                ),
                cls_probs=__class__._make_onehot(pred.category.id, self._num_classes),
                det_id=self._detection_counter.count(),
                img_id=img_id,
            )
            for tile, pred in zip(tiles, predictions_list)
        ]

    @staticmethod
    def _make_onehot(index: int, num_classes: int) -> np.ndarray:
        onehot = np.zeros(num_classes)
        onehot[index] = 1
        return onehot


class Counter:
    def __init__(self, init_val: int = 0):
        self._val = init_val

    def count(self):
        self._val += 1
        return self._val
