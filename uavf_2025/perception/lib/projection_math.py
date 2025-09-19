import numpy as np
from typing import Tuple
from perception.types import CameraPose
from shared.types import Pose
from perception.lib.util import pose_to_cv, CV_WORLD_FRAME_TO_WORLD_FRAME_MAT
import cv2


def project(
    pts3: np.ndarray, camera_pose: CameraPose, frame_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project the given 3D points in world coordinates into the specified camera.
    Parameters
    ----------
    pts3 : 2D torch.Tensor (dtype=float)
        Coordinates of N points stored in a tensor of shape (3, N).
    camera_pose : CameraPose
        Object containing the camera's rotation matrix, position, and focal length.
    frame_size : Tuple[int, int]
        The size of the image frame (width, height).
    Returns
    -------
    pts2 : 2D torch.Tensor (dtype=float)
        Image coordinates of N points stored in a tensor of shape (N,2).
    in_frame_arr : torch.Tensor
        Tensor of shape (N,) containing 1 if the point is inside the frame, otherwise 0.
    """
    rvec, tvec = pose_to_cv(Pose(camera_pose.position, camera_pose.rotation))
    intrinsics = np.array(
        [
            [camera_pose.focal_len_px, 0, frame_size[0] // 2],
            [0, camera_pose.focal_len_px, frame_size[1] // 2],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    image_points = cv2.projectPoints(
        pts3.T @ CV_WORLD_FRAME_TO_WORLD_FRAME_MAT, rvec, tvec, intrinsics, np.zeros(5)
    )[0].reshape((-1, 2))
    assert image_points.shape[0] == pts3.shape[1]
    assert image_points.shape[1] == 2
    return image_points, np.ones(image_points.shape[0])
