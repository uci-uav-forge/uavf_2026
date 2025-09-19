import logging
from perception.types import Image, CameraPose
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

import numpy as np
import cv2
from pathlib import Path

from line_profiler import profile

COVERAGE_WIDTH = 1000


def pixel_to_world_coords(x: int, y: int, image: Image, camera_pose: CameraPose):
    w, h = image.width, image.height
    focal_len = camera_pose.focal_len_px

    camera_position, rot_transform = camera_pose.position, camera_pose.rotation
    # the vector pointing out the camera at the target, if the camera was facing positive Z

    positive_y_direction = np.cross(np.array([1, 0, 0]), np.array([0, -1, 0]))

    initial_direction_vector = (
        focal_len * np.array([1, 0, 0])
        + (x - w / 2) * np.array([0, -1, 0])
        + (y - h / 2) * positive_y_direction
    )

    # rotate the vector to match the camera's rotation
    rotated_vector = rot_transform.apply(initial_direction_vector)

    # solve camera_pos + t*rotated_vector = [x,ground_coord,z] = target_position
    t = (0 - camera_position[2]) / rotated_vector[2]
    target_position = camera_position + t * rotated_vector
    return target_position


class DumbMapperSystem:
    """Mapper interface for perception.py"""

    LOG_MSG_PREFIX = "[Mapping/MapperSystem]: "

    def __init__(
        self,
        console_logger: logging.Logger | None,
        logging_path: Path | None,
        roi_coords_local: np.ndarray,
    ):
        """
        roi_coords_local needs to be the perimeter of the region of interest to be mapped,
        with shape (3, 2).

        For the purposes of getting an image back aligned to the ROI, the coordinates
        will be treated like (top_left, top_right, bottom_left)

        We assume flat ground with z=0. Units will be meters.
        """

        self._console_logger = (
            console_logger
            if console_logger is not None
            else logging.Logger("dumb_mapper_default_logger", logging.ERROR)
        )
        self._logging_path = logging_path
        if self._logging_path is not None:
            self._logging_path.mkdir(parents=True, exist_ok=True)
        self._roi_coords_local = roi_coords_local
        self._console_logger.info(f"Using ROI {roi_coords_local.tolist()}")
        top_left, top_right, bottom_left = roi_coords_local
        horizontal_dist = np.linalg.norm(top_left - top_right)
        vertical_dist = np.linalg.norm(top_left - bottom_left)
        area_meters = vertical_dist * horizontal_dist
        max_area = 4000 * 3000
        scale_factor = np.sqrt(max_area / area_meters)
        self.dimensions = (
            int(vertical_dist * scale_factor),
            int(horizontal_dist * scale_factor),
            3,
        )
        self.img = np.zeros(self.dimensions, dtype=np.uint8)
        self.coverage = np.zeros(self.dimensions[:2], dtype=np.uint16)

        self.sift = cv2.SIFT_create()
        self.matcher = cv2.FlannBasedMatcher()

    def optimize_homography(self, im1: np.ndarray, im2: np.ndarray, H: np.ndarray):
        return H

    @profile
    def stitch_with_optimization(self, image: Image, camera_pose: CameraPose) -> bool:
        rotation_angle_from_vertical = np.rad2deg(
            np.arccos(np.dot(camera_pose.rotation.apply([1, 0, 0]), [0, 0, -1]))
        )  # radians
        if rotation_angle_from_vertical > 10:
            self._console_logger.info("Filtering out due to too high angle")
            return False
        # step 1: get transform pixel coords
        # step 2: get cropped region of composite image and homography
        transform_pixel_coords = self.get_transform_pixel_coords(image, camera_pose)
        bbox_xyxy = np.array(
            [
                min(transform_pixel_coords[:, 0]),
                min(transform_pixel_coords[:, 1]),
                max(transform_pixel_coords[:, 0]),
                max(transform_pixel_coords[:, 1]),
            ]
        ).astype(np.int32)
        crop = self.img[bbox_xyxy[1] : bbox_xyxy[3], bbox_xyxy[0] : bbox_xyxy[2]]

        def pose_to_homography(cam_pose: CameraPose):
            transform_pixel_coords = self.get_transform_pixel_coords(image, cam_pose)
            return cv2.getPerspectiveTransform(
                np.array(
                    [
                        [0, 0],
                        [image.width, 0],
                        [0, image.height],
                        [image.width, image.height],
                    ],
                    dtype=np.float32,
                ),
                (transform_pixel_coords - bbox_xyxy[:2]).astype(np.float32),
            )

        homography = pose_to_homography(camera_pose)

        # step 3: optimize
        new_homography = self.optimize_homography(image.get_array(), crop, homography)

        # convert homography to camera pose
        # TODO: do this analytically. I'm too dumb to figure it out right now
        def residuals(x):
            pose = CameraPose(
                position=x[:3],
                rotation=Rotation.from_rotvec(x[3:]),
                focal_len_px=camera_pose.focal_len_px,
            )
            H = pose_to_homography(pose)
            return np.reshape(H - new_homography, (-1,))

        optim_res = least_squares(
            residuals,
            x0=camera_pose.position.tolist()
            + camera_pose.rotation.as_rotvec().tolist(),
        )
        new_camera_pose = CameraPose(
            position=optim_res.x[:3],
            rotation=Rotation.from_rotvec(optim_res.x[3:]),
            focal_len_px=camera_pose.focal_len_px,
        )

        # step 3: stitch with optimized pose
        return self.stitch(image, new_camera_pose)

    def get_transform_pixel_coords(self, image: Image, camera_pose: CameraPose):
        # figure out where map corners are in world coordinates
        # figure out where image corners are in world coordinates (on the ground)
        image_pixel_corners = np.array(
            [(0, 0), (image.width, 0), (0, image.height), (image.width, image.height)],
            dtype=np.float32,
        )
        image_world_corners = np.array(
            [
                pixel_to_world_coords(x, y, image, camera_pose)
                for x, y in image_pixel_corners
            ],
            dtype=np.float32,
        )
        # transform image world coords into map pixel coords

        top_left, top_right, bottom_left = self._roi_coords_local
        x_vector = top_right - top_left
        y_vector = bottom_left - top_left
        x_dist = np.linalg.norm(x_vector)
        y_dist = np.linalg.norm(y_vector)
        x_proj = np.dot(image_world_corners - top_left, x_vector / x_dist)
        y_proj = np.dot(image_world_corners - top_left, y_vector / y_dist)
        x_pixel = x_proj * self.img.shape[1] / x_dist
        y_pixel = y_proj * self.img.shape[0] / y_dist
        transform_pixel_coords = np.vstack([x_pixel, y_pixel]).T.astype(np.float32)
        return transform_pixel_coords

    @profile
    def stitch(self, image: Image, camera_pose: CameraPose) -> bool:
        """
        Add an image to the map using its camera pose
        """
        # TODO: un-hardcode principal point and incorporate dist coefficients
        # k_mat = np.array(
        #     [
        #         [camera_pose.focal_len_px, 0, 1920 // 2],
        #         [0, camera_pose.focal_len_px, 1080 // 2],
        #         [0, 0, 1],
        #     ]
        # )

        transform_pixel_coords = self.get_transform_pixel_coords(image, camera_pose)
        transform_bbox = np.array(
            [
                min(transform_pixel_coords[:, 0]),
                min(transform_pixel_coords[:, 1]),
                max(transform_pixel_coords[:, 0]),
                max(transform_pixel_coords[:, 1]),
            ]
        )

        offset_vector = transform_bbox[:2]
        transform_bbox = transform_bbox.astype(np.int32)

        transform_pixel_coords_translated = transform_pixel_coords - offset_vector

        crop_dims = (
            transform_bbox[2] - transform_bbox[0],
            transform_bbox[3] - transform_bbox[1],
        )

        image_pixel_corners = np.array(
            [
                (0, 0),
                (crop_dims[0], 0),
                (0, crop_dims[1]),
                (crop_dims[0], crop_dims[1]),
            ],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(
            image_pixel_corners, transform_pixel_coords_translated
        )
        ones_image = np.ones_like(image.get_array()) * 255
        warped_image = cv2.warpPerspective(image.get_array(), transform, crop_dims)
        mask_warp = cv2.warpPerspective(ones_image, transform, crop_dims)

        if transform_bbox[3] > self.img.shape[0]:
            warped_image = warped_image[
                : warped_image.shape[0] - (transform_bbox[3] - self.img.shape[0]), ...
            ]
            mask_warp = mask_warp[
                : mask_warp.shape[0] - (transform_bbox[3] - self.img.shape[0]), ...
            ]

        if transform_bbox[2] > self.img.shape[1]:
            warped_image = warped_image[
                :,
                : warped_image.shape[1] - (transform_bbox[2] - self.img.shape[1]),
                ...,
            ]
            mask_warp = mask_warp[
                :, : mask_warp.shape[1] - (transform_bbox[2] - self.img.shape[1]), ...
            ]

        if transform_bbox[1] < 0:
            warped_image = warped_image[-transform_bbox[1] :, ...]
            mask_warp = mask_warp[-transform_bbox[1] :, ...]
            transform_bbox[1] = 0

        if transform_bbox[0] < 0:
            warped_image = warped_image[:, -transform_bbox[0] :, ...]
            mask_warp = mask_warp[:, -transform_bbox[0] :, ...]
            transform_bbox[0] = 0

        if any(
            [
                mask_warp.shape[0] == 0,
                mask_warp.shape[1] == 0,
            ]
        ):
            self._console_logger.debug("Mask warp has no pixels, not stitching")
            return False

        # draw on coverage image
        coverage_coords = transform_pixel_coords
        to_be_added = np.zeros_like(self.coverage)
        cv2.fillConvexPoly(
            to_be_added,
            coverage_coords[(0, 1, 3, 2), :].astype(np.int32),
            color=1,
        )
        if np.sum(to_be_added) == 0:
            self._console_logger.debug("No new coverage to add, not stitching")
            return False

        self.img[
            transform_bbox[1] : transform_bbox[3], transform_bbox[0] : transform_bbox[2]
        ] = np.where(
            mask_warp == 255,
            warped_image,
            self.img[
                transform_bbox[1] : transform_bbox[3],
                transform_bbox[0] : transform_bbox[2],
            ],
        )
        self.coverage += to_be_added
        return True

    def clear(self) -> None:
        """
        Clears existing stored/aggregated map
        """
        self.img = np.zeros(self.dimensions)
        self.coverage = np.zeros(self.dimensions[:2])

    def get_map(self) -> np.ndarray:
        """
        Get the map of the region of interest, with the given resolution.
        The resolution is the size of each pixel in meters.
        """
        return np.array(self.img)

    def save_map(self) -> bool:
        if self._logging_path is None:
            return False
        fname = str(self._logging_path / "map.png")

        self._console_logger.info(self.LOG_MSG_PREFIX + f" Saving map [{fname}]")
        success = cv2.imwrite(fname, self.img)
        if success:
            self._console_logger.info(self.LOG_MSG_PREFIX + " Map saved")
            return True
        else:
            self._console_logger.error(self.LOG_MSG_PREFIX + " Map failed to save")
            return False

    def get_coverage_image(self):
        return cv2.resize(self.coverage, (1000, 1000))
