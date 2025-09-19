import numpy as np

from perception.camera.camera import ImageMetadata
from perception.types import Image
from shared.generation_utils import loop
from . import Camera


class DummyCamera(Camera):
    """
    Camera which replays the same list of images in a loop.
    """

    def __init__(
        self,
        images: list[Image],
        metadata: list[ImageMetadata],
    ):
        super().__init__()

        assert len(images) > 0, "Must provide at least one image to replay"
        assert len(images) == len(
            metadata
        ), "Must provide the same number of images and metadata"

        self._images_loop = loop(list(images))
        self._metadata_loop = loop(list(metadata))

        self._latest_metadata = metadata[0]

    def get_latest_image(self) -> Image[np.ndarray]:
        return next(self._images_loop)

    def take_image(self) -> Image:
        return self.get_latest_image()

    def get_metadata(self) -> ImageMetadata:
        self._latest_metadata = next(self._metadata_loop)
        return self._latest_metadata

    def get_focal_length_px(self) -> float:
        return self._latest_metadata.focal_len_px

    def _recording_worker(self):
        """
        Overridden to prevent recording from a replay camera.
        """
        pass

    def start_logging(self):
        """
        Overridden to prevent logging from a replay camera.
        """
        pass

    def stop_logging(self):
        """
        Overridden to prevent logging from a replay camera.
        """
        pass
