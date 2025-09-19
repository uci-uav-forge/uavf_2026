from pathlib import Path

from numpy import ndarray

from perception.types import Image
from .camera import Camera, ImageMetadata


class ReplayCamera(Camera):
    def __init__(self, source_dir: Path):
        super().__init__()

        # Reverse sort to pop from the end
        self._images = sorted(source_dir.glob("*.png"), reverse=True)
        self._metadata = sorted(source_dir.glob("*.json"), reverse=True)

        self._latest_metadata: ImageMetadata = ImageMetadata.load(self._metadata[0])

    def get_latest_image(self) -> Image[ndarray]:
        path = self._images.pop(0)
        return Image.from_file(path)  # type: ignore

    def take_image(self) -> Image:
        return self.get_latest_image()

    def get_metadata(self) -> ImageMetadata:
        path = self._metadata.pop(0)
        self._latest_metadata = ImageMetadata.load(path)
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
