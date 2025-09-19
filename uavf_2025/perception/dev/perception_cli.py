import numpy as np
import argparse
from pathlib import Path
from time import strftime, time
from perception import Perception
from shared.types import Pose
from perception.obstacle_tracking.types import Obstacle
from perception.lib.osd import draw_osd, draw_tracks_on_img
import cv2
import os

print(np.__file__, np.__version__)


def visualize_obstacles(drone_pose: Pose, obstacles: list[Obstacle]):
    canvas = np.zeros((1000, 1000, 3), dtype=np.uint8)

    # scale image such that it covers at least 5x5 meters, and always includes every obstacle with a 1 meter buffer
    scale = max(
        5,
        max(
            [obstacle.state[0] for obstacle in obstacles]
            + [obstacle.state[1] for obstacle in obstacles]
        )
        + 1,
    )
    cv2.putText(
        canvas,
        f"Scale: {scale}",
        (0, 25),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (255, 255, 255),
        thickness=2,
    )
    cv2.circle(canvas, (500, 500), int(100 / scale), (255, 255, 255), -1)
    look_offset_xy = drone_pose.rotation.apply(np.array([1, 0, 0]))[:2] * 5
    cv2.line(
        canvas,
        (500, 500),
        (int(500 + look_offset_xy[0]), int(500 - look_offset_xy[1])),
        (255, 255, 255),
        thickness=1,
    )

    # draw obstacles relative to drone pose
    for obstacle in obstacles:
        if np.any(np.isnan(obstacle.state)):
            continue  # really these should never be NaN, we need to investigate why this happens
        # draw circle at obstacle position
        cv2.circle(
            canvas,
            (
                500 + int((obstacle.state[0] - drone_pose.position[0]) * 500 / scale),
                500 - int((obstacle.state[1] - drone_pose.position[1]) * 500 / scale),
            ),
            2,
            (0, 0, 255),
            2,
        )
    cv2.imshow("obstacles", canvas)
    cv2.waitKey(1)


def main():
    parser = argparse.ArgumentParser(description="Perception system for UAVF 2025.")
    # add unnamed argument for camera source
    parser.add_argument(
        "camera_source",
        type=str,
        default="sim",
        help='Camera source for perception system.. e.g. "gazebo", "zr10", or "mock"',
    )
    parser.add_argument("--no-mapping", dest="mapping", action="store_false")
    parser.set_defaults(mapping=True)
    parser.add_argument("--benchmarking", dest="benchmarking", action="store_true")
    parser.set_defaults(benchmarking=False)
    parser.add_argument(
        "--testing-with-rosbag", dest="rosbag_testing", action="store_true"
    )
    parser.set_defaults(rosbag_testing=False)
    args = parser.parse_args()

    cam_source_str = args.camera_source
    print(f"Using camera source: {cam_source_str}")

    if cam_source_str == "sim":
        cam_source = Perception.CameraType.GAZEBO
    elif cam_source_str == "zr10":
        cam_source = Perception.CameraType.ZR10
    elif cam_source_str == "a8":
        cam_source = Perception.CameraType.A8
    else:
        raise ValueError(
            f"Invalid camera source: {cam_source_str}. Needs to be 'sim', 'zr10', or 'mock'."
        )

    do_run_avoidance = bool(os.getenv("RUN_AVOIDANCE", False))

    logs_path = Path(f"logs/nvme/{strftime('%Y-%m-%d_%H-%M')}")

    perception = Perception(
        cam_source.value,
        logs_path,
        enable_mapping=True,
        mapping_path=logs_path / "mapping",
        enable_benchmarking=args.benchmarking,
        enable_tracking=do_run_avoidance,
        rosbag_testing=args.rosbag_testing,
    )

    is_recording = True
    perception.start_recording()

    print(
        f'Current mode: {perception._mode}. Press "a" to set area scan, "t" to set target lock, "i" to set idle, "r" to toggle recording, "q" to quit.'
    )
    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("preview", 1024, 600)
    if do_run_avoidance:
        cv2.namedWindow("drone_det_debug", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("drone_det_debug", 1280, 720)
    while True:
        if perception.get_camera().recording:
            latest_img = perception.get_camera().get_latest_image()
        else:
            latest_img = perception.get_camera().take_image()
        if latest_img is None:
            if perception._backup_camera is not None:
                latest_img = (
                    perception._backup_camera.get_latest_image()
                    if perception._backup_camera.recording
                    else perception._backup_camera.take_image()
                )
                if latest_img is None:
                    print("no img from either cam")
                    continue
            else:
                print("no img on primary, and no backup cam.")
                continue

        latest_img = latest_img.get_array()
        to_display = latest_img.copy()
        try:
            drone_pose = perception._drone_pose_provider.get_pose_at_time(time())
            if perception._mode == Perception.Mode.AREA_SCAN:
                draw_tracks_on_img(
                    to_display,
                    perception.get_target_positions(),
                    perception._camera.get_world_pose(drone_pose),
                )
        except BufferError:
            continue
        drone_pose = perception._drone_pose_provider.get_pose_at_time(time())
        cam_pose = perception._camera.get_world_pose(drone_pose)
        draw_osd(
            to_display,
            {
                "timestamp": time(),
                "mode": perception._mode.value,
                "position": list(cam_pose.position),
            },
            cam_pose,
            False,
            drone_pose,
        )
        if perception._mode == Perception.Mode.AREA_SCAN:
            draw_tracks_on_img(to_display, perception.get_target_positions(), cam_pose)

        if do_run_avoidance:
            visualize_obstacles(drone_pose, perception.get_obstacles())
            drone_det_debug = perception.get_drone_track_debug_img()
            if drone_det_debug is not None:
                cv2.imshow("drone_det_debug", drone_det_debug)

        cv2.imshow("preview", to_display)
        key = cv2.waitKey(1)
        if not ord("a") <= key <= ord("z"):
            continue
        user_input = chr(key)
        # user_input = input()
        if len(user_input) > 1 or user_input < "a" or user_input > "z":
            continue
        if user_input == "a":
            perception.set_area_scan(None, latest_img)
        elif user_input == "t":
            perception.set_target_lock(0)
        elif user_input == "i":
            perception.set_idle()
        elif user_input == "d":
            perception.get_camera().point_down()
        elif user_input == "f":
            perception.get_camera().point_center()
        elif user_input == "q":
            print("quitting")
            perception.cleanup()
            break
        elif user_input == "r":
            if is_recording:
                perception.stop_recording()
                is_recording = False
                print("Stopped recording.")
            else:
                perception.start_recording()
                is_recording = True
                print("Started recording.")
        else:
            print(f"Invalid input: {user_input}")
            continue
        print(
            f'Current mode: {perception._mode}. Press "a" to set area scan, "t" to set target lock, "i" to set idle, "r" to toggle recording, "q" to quit.'
        )


if __name__ == "__main__":
    main()
