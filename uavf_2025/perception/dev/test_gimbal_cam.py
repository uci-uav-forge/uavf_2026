from perception.camera import (
    A8Camera,
    ZR10Camera,
    GazeboGimballedCamera,
    make_gimballed_camera,
)
from perception.lib.osd import draw_osd
from shared.types import Pose
from time import time, strftime
from pathlib import Path
import cv2
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perception system for UAVF 2025.")
    # add unnamed argument for camera source
    parser.add_argument(
        "camera_source",
        type=str,
        default="sim",
        help='Camera source for perception system.. e.g. "gazebo", "zr10", or "mock"',
    )
    args = parser.parse_args()

    cam_source_str = args.camera_source
    print(f"Using camera source: {cam_source_str}")

    if cam_source_str == "sim":
        cam_source = GazeboGimballedCamera
    elif cam_source_str == "zr10":
        cam_source = ZR10Camera
    elif cam_source_str == "a8":
        cam_source = A8Camera
    else:
        print("Using auto gimballed camera function")
        cam_source = make_gimballed_camera

    logs_base = Path("logs/nvme")
    time_dir = Path(strftime("%Y-%m-%d_%H-%M"))
    logs_path = logs_base / "gimbal_test" / time_dir
    camera = cam_source(logs_path)
    camera.set_absolute_zoom(1)

    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("preview", 1024, 600)

    yaw = 0
    pitch = 0
    while True:
        frame = camera.take_image()
        if frame is None:
            continue
        to_display = np.ascontiguousarray(frame.get_array(), dtype=np.uint8)
        cam_pose = camera.get_world_pose(Pose.identity())
        draw_osd(
            to_display,
            {"timestamp": time(), "attitude": camera.get_attitude()},
            cam_pose,
            False,
            None,
        )
        cv2.imshow("preview", to_display)
        key = cv2.waitKey(1)
        if ord("1") <= key <= ord("9"):
            print(key - ord("0"))
            camera.set_absolute_zoom(key - ord("0"))
        elif key == ord("d"):  # down
            camera.set_absolute_position(0, -90)
            yaw = 0
            pitch = -90
        elif key == ord("c"):  # center
            camera.set_absolute_position(0, 0)
            yaw = 0
            pitch = 0
        elif key == ord("l"):  # left
            camera.set_absolute_position(135, 0)
            yaw = 135
            pitch = 0
        elif key == ord("r"):  # right
            camera.set_absolute_position(-135, 0)
            yaw = -135
            pitch = 0
        elif key == ord("p"):  # print
            print(camera.get_attitude())
            print(camera.get_attitude_speed())
        elif key == ord("s"):  # save (record, but R was already used)
            if camera.recording:
                camera.stop_recording()
            else:
                camera.start_recording()
        elif key == ord("q"):
            break
        elif key == 81:  # left
            yaw += 1
            camera.set_absolute_position(yaw, pitch)
        elif key == 82:  # up
            pitch += 1
            camera.set_absolute_position(yaw, pitch)
        elif key == 83:  # right
            yaw -= 1
            camera.set_absolute_position(yaw, pitch)
        elif key == 84:  # down
            pitch -= 1
            camera.set_absolute_position(yaw, pitch)
        else:
            continue
