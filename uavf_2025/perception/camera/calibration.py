from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

import cv2
import numpy as np


# TODO: Implement Aruco pattern calibration
class CameraCalibration(NamedTuple):
    matrix: np.ndarray
    distortion: np.ndarray

    def undistort(self, image: np.ndarray) -> np.ndarray:
        return cv2.undistort(image, self.matrix, self.distortion)

    def to_json(self):
        return {"matrix": self.matrix.tolist(), "distortion": self.distortion.tolist()}

    def save(self, path: Path | str):
        with open(path, "w") as f:
            json.dump(self.to_json(), f)
        print("Saved CameraCalibration to", path)

    @staticmethod
    def load(path: Path | str) -> "CameraCalibration":
        with open(path, "r") as f:
            data = json.load(f)
            return CameraCalibration(
                np.array(data["matrix"]), np.array(data["distortion"])
            )

    @staticmethod
    def from_chessboards(
        images: Iterable[np.ndarray],
        corners_grid: tuple[int, int] = (9, 6),
        preview: bool = False,
        alpha: float = 0,
    ) -> "CameraCalibration":
        """
        NOTE: `corners_grid` is the number of corners in the chessboard, not the number of squares.
        """
        object_points: list[np.ndarray] = []
        image_points: list[np.ndarray] = []

        img_size: tuple[int, int] | None = None

        good = 0
        for index, image in enumerate(images):
            if len(image.shape) != 2:
                print(f"Image {index} is not grayscale. Skipping")
                continue

            if img_size is None:
                img_size = (image.shape[1], image.shape[0])

            img_pts = CameraCalibration._find_chessboard_corners(
                image, corners_grid, preview
            )

            if img_pts is None:
                print(f"Failed to find corners in image {index}. Skipping.")
                continue

            object_points.append(CameraCalibration._make_obj_points(corners_grid))
            image_points.append(img_pts)

            good += 1

        print(f"CameraCalibration: Found {good} good images")

        if img_size is None:
            raise ValueError("No valid images found.")

        return CameraCalibration.from_correspondences(
            object_points, image_points, img_size, alpha
        )

    @staticmethod
    def from_correspondences(
        object_points: Sequence[np.ndarray],
        image_points: Sequence[np.ndarray],
        img_size: Sequence[int],  # (width, height)
        alpha: float = 0,
    ) -> "CameraCalibration":
        _, matrix, distortion, _, _ = cv2.calibrateCamera(
            object_points,
            image_points,
            img_size,
            None,  # type: ignore
            None,  # type: ignore
        )

        matrix, _ = cv2.getOptimalNewCameraMatrix(matrix, distortion, img_size, alpha)

        return CameraCalibration(matrix, distortion)

    @staticmethod
    def _make_obj_points(grid_size: tuple[int, int]):
        objp = np.zeros((grid_size[1] * grid_size[0], 3), np.float32)
        objp[:, :2] = np.mgrid[0 : grid_size[0], 0 : grid_size[1]].T.reshape(-1, 2)

        return objp

    @staticmethod
    def _find_chessboard_corners(
        image: np.ndarray, corners_grid: tuple[int, int], preview: bool
    ):
        success, corners = cv2.findChessboardCorners(image, corners_grid)

        if not success:
            return None

        img_pts = cv2.cornerSubPix(
            image,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )

        if preview:
            cv2.drawChessboardCorners(image, corners_grid, corners, success)
            cv2.imshow("Detected Chessboard", image)
            cv2.waitKey(0)

        return img_pts
