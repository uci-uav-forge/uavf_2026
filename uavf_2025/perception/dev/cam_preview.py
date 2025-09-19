import cv2
import numpy as np
from perception.camera import Camera, GazeboCamera, ZR10Camera, A8Camera
from perception.camera.usb_cam import USBCam
from perception.camera.csi_cam import CSICam
from pathlib import Path
from time import strftime
import os

if __name__ == "__main__":
    camera_selection = input("""
        Enter camera selection:
        0: usb camera
        1: zr10 gimbal
        2: CSI camera
        3: a8 gimbal
        4: gazebo front cam
        5: gazebo gimbal cam
    """).strip()
    show_gui = input("GUI? (y,n,r) (r=reduced, 64x36 image)")
    logs_base = Path("logs/nvme")
    time_dir = Path(strftime("%Y-%m-%d_%H-%M"))

    if camera_selection == "0":
        logs_path = logs_base / "usb" / time_dir
        camera: Camera = USBCam(logs_path)
    elif camera_selection == "1":
        logs_path = logs_base / "zr10" / time_dir
        camera: Camera = ZR10Camera(logs_path)
    elif camera_selection == "2":
        logs_path = logs_base / "csi" / time_dir
        camera: Camera = CSICam(logs_path, CSICam.ResolutionOption.R4K)
    elif camera_selection == "3":
        logs_path = logs_base / "a8" / time_dir
        camera: Camera = A8Camera(logs_path)
    elif camera_selection == "4":
        logs_path = logs_base / "gz_front" / time_dir
        camera = GazeboCamera(
            logs_path,
            "/world/map/model/iris/link/avoidance_cam_front_link/sensor/camera/image",
        )
    elif camera_selection == "5":
        logs_path = logs_base / "gz_gimbal" / time_dir
        camera = GazeboCamera(logs_path)
    else:
        raise ValueError("Invalid selection")

    camera.start_recording()
    if show_gui == "y" or show_gui == "r":
        res = (1024, 600) if show_gui != "r" else (32, 18)
        placeholder = np.zeros(res)
        cv2.putText(
            placeholder,
            "No image",
            (10, 10),
            cv2.FONT_HERSHEY_COMPLEX,
            2,
            (255, 255, 255),
        )
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("test", *res)
    while True:
        if camera.recording:
            frame = camera.get_latest_image()
        else:
            frame = camera.take_image()
        if show_gui == "y" or show_gui == "r":
            to_display = frame.get_array() if frame is not None else placeholder
            cv2.imshow("test", to_display)
            key = cv2.waitKey(1)

            if key == ord("q"):
                break
        else:
            print(f"files in folder {len(os.listdir(logs_path))}", end="\r")
