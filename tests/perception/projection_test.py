from perception.lib.projection_math import project, draw_points_on_img
from perception.types import CameraPose
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
from shared.types import Pose
import cv2
import json


def test_project_coord_no_camera_rot():
    test_pt = torch.Tensor([[1, 1], [1, -1], [1, -1]])
    cam_pose = CameraPose(np.zeros(3), R.identity(), 1920)
    frame_size = (1920, 1080)
    pt2, in_frame = project(test_pt, cam_pose, frame_size)


def test_project_coord_cam_yaw_90():
    test_pt = torch.Tensor([[1, 1], [1, -1], [1, -1]])
    cam_pose = CameraPose(
        np.array([1, 0, 0]), R.from_euler("z", 90, degrees=True), 1920
    )
    frame_size = (1920, 1080)
    pt2, in_frame = project(test_pt, cam_pose, frame_size)


def test_on_data():
    data = json.load(open("tests/perception/804.json", "r"))

    points_grid_center = torch.tensor([[5, -3, 0]]).T
    xs, ys = torch.meshgrid(torch.arange(-100, 101, 2), torch.arange(-100, 101, 2))
    points_grid = (
        torch.stack(
            [
                xs.flatten(),
                ys.flatten(),
                torch.zeros_like(xs.flatten()),
            ]
        )
        + points_grid_center
    )

    test_pts = points_grid

    # test_pts = torch.Tensor([
    #     [-1, 14, 0],
    #     [0, 14, 0],
    #     [-2, 14, 0],
    #     [-1, 13, 0],
    #     [-1, 15, 0],
    #     [-1, 14, -1],
    #     [-1, 14, 1],
    # ]).T

    closest_pose = Pose.from_dict(
        json.load(open("tests/perception/1731172134.120311488.json", "r"))
    )

    cam_position = np.array(data["relative_pose"]["position"])
    cam_rotation = R.from_quat(data["relative_pose"]["rotation_quat"])

    combined_pose = CameraPose(
        cam_position,
        cam_rotation,
        data["focal_len_px"],
    ).with_pose(closest_pose)

    img = cv2.imread("tests/perception/804.jpg")
    draw_points_on_img(img, combined_pose, test_pts)

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 1280, 720)
    cv2.imshow("image", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    test_on_data()
