import torch
from perception.types import CameraPose, Pose
from typing import Tuple


def coordinate_transform(pts3: torch.Tensor, zero_pose: Pose) -> torch.Tensor:
    """
    Transform the given 3D points from world coordinates to zero the given pose.
    """
    rotation_matrix = torch.tensor(
        zero_pose.rotation.as_matrix().T, dtype=pts3.dtype, device=pts3.device
    )

    camera_position = torch.tensor(
        zero_pose.position, dtype=pts3.dtype, device=pts3.device
    )

    translated = (pts3.T - camera_position).T

    # Apply rotation
    return rotation_matrix @ translated


def project(
    pts3: torch.Tensor, camera_pose: CameraPose, frame_size: Tuple[int, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project the given 3D points in world coordinates into the specified camera.

    Parameters
    ----------
    pts3 : 2D torch.Tensor (dtype=float)
        Coordinates of N points stored in a tensor of shape (3, N).

    camera_pose : CameraPose
        Object containing the camera's rotation matrix, position, and focal length.

    frame_size : Tuple[int, int]
        The size of the image frame (height, width).

    Returns
    -------
    pts2 : 2D torch.Tensor (dtype=float)
        Image coordinates of N points stored in a tensor of shape (2, N).

    in_frame_arr : torch.Tensor
        Tensor of shape (N,) containing 1 if the point is inside and in front of the frame, otherwise 0.
    """
    assert pts3.shape[0] == 3

    height, width = frame_size

    cam_center_2d = torch.tensor(
        [width / 2, height / 2], dtype=torch.float32, device=pts3.device
    ).view(2, 1)

    # Points' positions in camera coordinates
    cam_points_3d = coordinate_transform(pts3, camera_pose.as_pose())

    # Project to the image plane
    projected = torch.as_tensor(camera_pose.focal_len_px) * (
        cam_points_3d / cam_points_3d[2]
    )

    # Add the camera center to get final image coordinates
    pts2 = projected[0:2, :] + cam_center_2d

    assert pts2.shape[1] == pts3.shape[1]
    assert pts2.shape[0] == 2

    x_coords = pts2[0]  # Series of x-coordinates
    y_coords = pts2[1]  # Series of y-coordinates

    in_front = cam_points_3d[2] > 0  # Check if points are in front of the frustrum

    # Ensure that points are within the frame bounds
    point_is_in_frame = torch.logical_and(
        torch.logical_and(x_coords >= 0, x_coords <= width),
        torch.logical_and(y_coords >= 0, y_coords <= height),
    )

    point_is_valid = torch.logical_and(point_is_in_frame, in_front)

    return (pts2, point_is_valid)
