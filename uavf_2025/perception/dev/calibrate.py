import cv2 as cv

from typing import NamedTuple

import numpy as np
import os

from perception.camera import Camera, GazeboCamera, ZR10Camera, A8Camera
from perception.camera.gimballed_camera import GimballedCamera
from perception.camera.usb_cam import USBCam
from perception.camera.csi_cam import CSICam
from time import strftime
from pathlib import Path
from typing import Any


class BoardDetectionResults(NamedTuple):
    charuco_corners: Any
    charuco_ids: Any
    aruco_corners: Any
    aruco_ids: Any


class PointReferences(NamedTuple):
    object_points: Any
    image_points: Any


class CameraCalibrationResults(NamedTuple):
    repError: float
    camMatrix: Any
    distcoeff: Any
    rvecs: Any
    tvecs: Any


SQUARE_LENGTH = 500
MARKER_LENGHT = 300
NUMBER_OF_SQUARES_VERTICALLY = 11
NUMBER_OF_SQUARES_HORIZONTALLY = 8

charuco_marker_dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
charuco_board = cv.aruco.CharucoBoard(
    size=(NUMBER_OF_SQUARES_HORIZONTALLY, NUMBER_OF_SQUARES_VERTICALLY),
    squareLength=SQUARE_LENGTH,
    markerLength=MARKER_LENGHT,
    dictionary=charuco_marker_dictionary,
)

cam_mat = np.array([[1000, 0, 1920 // 2], [0, 1000, 1080 // 2], [0, 0, 1]])
dist_coeffs = np.zeros(5)

total_object_points = []
total_image_points = []
total_images_used = []

LIVE = bool(os.getenv("LIVE", True))

if LIVE:
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
    cv.namedWindow("calib", cv.WINDOW_NORMAL)
    cv.resizeWindow("calib", (1600, 900))

index = 0
imgs_path = logs_path / "calib_imgs"
imgs_path.mkdir(exist_ok=True)
images = sorted(list(imgs_path.glob("*.png")))

det_results: list[BoardDetectionResults] = []

while True:
    if LIVE:
        if isinstance(camera, GimballedCamera):
            camera.do_autofocus()
        img_bgr = camera.take_image().get_array()
    else:
        if index == len(images):
            break
        img_bgr = cv.imread(f"{images[index]}")
        index += 1
        print(f"Processing image {index}/{len(images)}")

    img_debug = img_bgr.copy()

    img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    charuco_detector = cv.aruco.CharucoDetector(charuco_board)
    detection_results = BoardDetectionResults(*charuco_detector.detectBoard(img_gray))

    if (
        detection_results.charuco_corners is not None
        and len(detection_results.charuco_corners) > 4
    ):
        det_results.append(detection_results)
        point_references = PointReferences(
            *charuco_board.matchImagePoints(
                detection_results.charuco_corners, detection_results.charuco_ids
            )
        )

        ret, rvecs, tvecs = cv.solvePnP(
            point_references.object_points,
            point_references.image_points,
            cam_mat,
            dist_coeffs,
            flags=cv.SOLVEPNP_IPPE,
        )
        if ret:
            reproj = cv.projectPoints(
                point_references.object_points, rvecs, tvecs, cam_mat, dist_coeffs
            )[0].squeeze()

            for pt in point_references.image_points:
                cv.circle(
                    img_debug, tuple(pt.squeeze().astype(int)), 10, (255, 0, 0), -1
                )

            for pt in reproj:
                cv.circle(img_debug, tuple(pt.astype(int)), 7, (0, 0, 255), -1)
    else:
        point_references = None

    if LIVE:
        cv.imshow("calib", img_debug)
        key = cv.waitKey(1)
    else:
        key = 1
    shape = img_bgr.shape[:2]
    if (not LIVE) or key == 13:
        print(len(total_images_used))
        if point_references is not None and len(point_references.object_points) > 4:
            total_object_points.append(point_references.object_points)
            total_image_points.append(point_references.image_points)
            total_images_used.append(img_bgr)
        if (
            LIVE and len(total_images_used) > 5 and len(total_images_used) % 5 == 0
        ) or (not LIVE and index == len(images)):
            calibration_results = CameraCalibrationResults(
                *cv.calibrateCamera(
                    total_object_points,
                    total_image_points,
                    shape,
                    None,  # type: ignore
                    None,  # type: ignore
                    flags=cv.CALIB_THIN_PRISM_MODEL,  # + cv.CALIB_USE_INTRINSIC_GUESS
                )
            )

            print(calibration_results.repError)
            print(calibration_results.camMatrix)
            print(calibration_results.distcoeff)
            pass
        if LIVE:
            cv.imwrite(f'{imgs_path}/{len(list(imgs_path.glob("*.png")))}.png', img_bgr)

    elif key == ord("q"):
        break


res = cv.aruco.calibrateCameraCharucoExtended(
    [d.charuco_corners for d in det_results],
    [d.charuco_ids for d in det_results],
    charuco_board,
    (1848, 3280),
    cam_mat,
    dist_coeffs,  # type: ignore
    # None,
    # flags=cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL + cv.CALIB_USE_INTRINSIC_GUESS,
    flags=cv.CALIB_RATIONAL_MODEL
    + cv.CALIB_USE_INTRINSIC_GUESS
    + cv.CALIB_THIN_PRISM_MODEL
    + cv.CALIB_TILTED_MODEL,
    criteria=(cv.TERM_CRITERIA_EPS & cv.TERM_CRITERIA_COUNT, 10000, 1e-9),
)

print(res[0])  # repError
print(res[1].__repr__())  # camMatrix
print(res[2].__repr__())  # distcoeff
