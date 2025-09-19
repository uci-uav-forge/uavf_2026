# OSD = on screen display
from perception.types import CameraPose
from perception.odlc.target_tracking import TargetTracker
from shared.types import Pose
import numpy as np
from .projection_math import project
import cv2
import json
import warnings

from typing import TypeVar
from line_profiler import profile


def pprint_prep_json_floats(d: dict, digits=2) -> dict:
    """
    Recursively traverses a dictionary and rounds all floats to a specified number of digits.
    """

    recursive_round_param_t = TypeVar("recursive_round_param_t", float, dict, list)

    def recursive_round(v: recursive_round_param_t) -> recursive_round_param_t:
        if isinstance(v, float):
            return round(v, digits)
        if isinstance(v, dict):
            return {k: recursive_round(v) for k, v in v.items()}
        if isinstance(v, list):
            return [recursive_round(v) for v in v]
        return v

    return recursive_round(d)


@profile
def draw_points_on_img(
    img: np.ndarray,
    cam_pose: CameraPose,
    points_3d: np.ndarray,
    shape="triangle",
    color_bgr=(0, 0, 255),
    labels: list[str] | None = None,
):
    projected_points, in_frame_arr = project(
        points_3d, cam_pose, (img.shape[1], img.shape[0])
    )

    def normalize_point(p):
        return (p - points_3d.min()) / (points_3d.max() - points_3d.min())

    # Draw circles on the img based on the projected points and size
    for (x, y), in_frame, point_3d, label in zip(
        projected_points,
        in_frame_arr,
        points_3d.T,
        labels if labels is not None else [None] * len(points_3d.T),
    ):
        r, g, b = normalize_point(point_3d) * 255
        if in_frame:
            if shape == "triangle":
                cv2.drawMarker(
                    img,
                    (int(x), int(y)),
                    (int(b), int(g), int(r)) if color_bgr is None else color_bgr,
                    markerType=cv2.MARKER_TRIANGLE_UP,
                    markerSize=30,
                    thickness=4,
                )
            elif shape == "circle":
                cv2.circle(
                    img,
                    (int(x), int(y)),
                    6,
                    (
                        (
                            int(b),
                            int(g),
                            int(r),
                        )
                        if color_bgr is None
                        else color_bgr
                    ),
                    -1,
                )
            elif shape is not None:
                warnings.warn(f"Unknown shape, drawing nothing: {shape}")

            (
                cv2.putText(
                    img,
                    f"{point_3d[0]:.1f}, {point_3d[1]:.1f}, {point_3d[2]:.1f}",
                    (int(x), int(y)),
                    0,
                    0.5,
                    (200, 200, 200),
                    2,
                )
                if label is None
                else cv2.putText(
                    img,
                    label,
                    (int(x), int(y)),
                    0,
                    1,
                    (255, 255, 255),
                    2,
                )
            )


@profile
def draw_tracks_on_img(
    img: np.ndarray, tracks: list[TargetTracker.Track], cam_pose: CameraPose
):
    if len(tracks) > 0:
        all_track_coords = np.array([t.position for t in tracks]).T
        labels = [
            f"{t.id}|{len(t.contributing_detections)} ({t.position[0]:.1f},{t.position[1]:.1f})"
            for t in tracks
        ]
        for track in tracks:
            draw_axis_on_img(img, cam_pose, track.position)
        # draw the text (the function name is a bad description of what's happening here)
        draw_points_on_img(img, cam_pose, all_track_coords, labels=labels, shape=None)


@profile
def draw_axis_on_img(
    img: np.ndarray, cam_pose: CameraPose, coords: np.ndarray, length: float = 1
):
    """
    Draws the x, y, z axes on the image based on the camera pose.
    Args:
        img: The image to draw on.
        cam_pose: The camera pose.
        coords: The coordinates of the axes in 3D space.
        length: The length of the axes to draw.
    """
    # Define colors for each axis
    colors = {
        "x": (0, 0, 255),  # Red for X axis
        "y": (0, 255, 0),  # Green for Y axis
        "z": (255, 0, 0),  # Blue for Z axis
    }

    # Create points for each axis
    axes_points = {
        "x": coords + np.array([length, 0, 0]),
        "y": coords + np.array([0, length, 0]),
        "z": coords + np.array([0, 0, length]),
    }

    # Project points to image plane
    axes_to_project_points = {}

    proj_points, _ = project(
        np.array(list(axes_points.values())).T, cam_pose, (img.shape[1], img.shape[0])
    )
    for axis, point in zip(axes_points.keys(), proj_points):
        axes_to_project_points[axis] = point
    coords_projected, _ = project(
        coords.reshape((3, -1)), cam_pose, (img.shape[1], img.shape[0])
    )
    coords_2d = coords_projected[0]
    # Draw axes on the image
    for axis, point in axes_to_project_points.items():
        if axis in colors:
            cv2.line(
                img,
                (int(coords_2d[0]), int(coords_2d[1])),
                (int(point[0]), int(point[1])),
                colors[axis],
                2,
            )


@profile
def draw_osd(
    img: np.ndarray,
    img_metadata: dict,
    camera_pose: CameraPose | None,
    draw_ground_grid=False,
    drone_pose: Pose | None = None,
):
    """
    Draws an on-screen display on an image to help with debugging. The OSD includes arbitrary metadata as
    text in the top-left, and reticles and a crosshair if camera_pose is provided. Optionally draws an augmented reality
    grid on the ground plane if draw_ground_grid is True.
    """
    data_text = json.dumps(pprint_prep_json_floats(img_metadata), indent=2)
    data_text = data_text.replace('"', "")
    for i, line in enumerate(data_text.split("\n")[1:-1]):
        cv2.putText(img, line, (0, 30 + 25 * i), 0, 1, (255, 255, 255), 2)

    if camera_pose is None:
        return

    if draw_ground_grid:
        points_grid_center = np.array([[0, 0, 0]]).T
        xs, ys = np.meshgrid(np.arange(-100, 101, 5), np.arange(-100, 101, 5))
        points_grid = (
            np.stack(
                [
                    xs.flatten(),
                    ys.flatten(),
                    np.zeros_like(xs.flatten()),
                ]
            )
            + points_grid_center
        )

        draw_points_on_img(
            img, camera_pose, points_grid, shape="circle", color_bgr=None
        )

    body_pose = Pose.identity() if drone_pose is None else drone_pose

    angles = np.linspace(0, np.pi / 2, 10)
    # draw lines from 0 to -90 degree pitch in drone body frame
    left_points = np.array(
        [
            body_pose.position
            + body_pose.rotation.apply(
                np.array(
                    [np.sin(theta), 0.01 + 0.15 * theta / np.pi / 2, -np.cos(theta)]
                )
            )
            for theta in angles
        ]
    )

    right_points = np.array(
        [
            body_pose.position
            + body_pose.rotation.apply(
                np.array(
                    [np.sin(theta), -(0.01 + 0.15 * theta / np.pi / 2), -np.cos(theta)]
                )
            )
            for theta in angles
        ]
    )
    left_projected_points, left_in_frame_arr = project(
        left_points.T, camera_pose, (img.shape[1], img.shape[0])
    )

    right_projected_points, right_in_frame_arr = project(
        right_points.T, camera_pose, (img.shape[1], img.shape[0])
    )

    for left_point, right_point, left_in_frame, right_in_frame, angle in zip(
        left_projected_points,
        right_projected_points,
        left_in_frame_arr,
        right_in_frame_arr,
        angles,
    ):
        if left_in_frame and right_in_frame:
            cv2.line(
                img,
                (int(left_point[0]), int(left_point[1])),
                (int(right_point[0]), int(right_point[1])),
                (255, 255, 255),
                3,
            )
            cv2.putText(
                img,
                f"{int(round(np.degrees(angle)))-90}",
                (int(right_point[0]) + 5, int(right_point[1])),
                0,
                1,
                (255, 255, 255),
                2,
            )

    # draw crosshair in center of image
    cv2.circle(
        img,
        (img.shape[1] // 2, img.shape[0] // 2),
        20,
        (255, 255, 255),
        2,
    )
    for dx, dy in [(0, 10), (0, -10), (10, 0), (-10, 0)]:
        cv2.line(
            img,
            (img.shape[1] // 2 + 2 * dx, img.shape[0] // 2 + 2 * dy),
            (img.shape[1] // 2 + 4 * dx, img.shape[0] // 2 + 4 * dy),
            (255, 255, 255),
            2,
        )
