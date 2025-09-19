import threading
from enum import Enum
from pathlib import Path
from time import sleep, time
from typing import Optional

import numpy as np
import torch
from line_profiler import profile

from perception.camera.siyi import A8Camera, ZR10Camera
from perception.camera.gazebo import GazeboGimballedCamera, GazeboCamera
from perception.camera.gimballed_camera import GimballedCamera
from perception.camera.auto_gimballed_cam import make_gimballed_camera
from perception.lib.util import create_console_logger, ProcessMemLogger
from perception.mapping.dumb_mapper import DumbMapperSystem
from perception.obstacle_tracking import (
    ObstacleTracker,
    ParticleFilter,
    YoloObstacleDetector,
)
from perception.odlc.target_tracking import TargetTracker
from shared import DronePoseProvider
from shared.git_info import get_git_diff, get_git_info

# from .camera import SiyiCam
import os


class Perception:
    """
    Perception operates in one of three modes:

    Idle mode: Perception completely controls the camera. It points the camera straight
    down if we have enough altitude, and otherwise points the camera forward.

    Area scan mode: Perception relinquishes control of the camera's gimbal and zoom to
    GNC via the get_camera() method. Perception then controls the shutter and updates
    the target tracker with the images it receives. GNC at any point can call get_target_positions()
    to get the estimated positions of the targets in the region of interest.

    Target lock mode: Perception takes complete control of the camera and gimbal to keep it pointed
    at the target with the given ID. It will also zoom in on the target as much as it can, and
    refine its estimate of the target's position and classification. GNC can call get_target_positions()
    and get the estimated position of the target with the given ID as the estimate is being updated.
    """

    class Mode(Enum):
        IDLE = "Idle"
        AREA_SCAN = "Area Scan"
        TARGET_LOCK = "Target Lock"

    class GimbalFlipState(Enum):
        LOOKING_DOWN = 0
        LOOKING_FORWARD = 1

    class CameraType(Enum):
        A8 = A8Camera
        ZR10 = ZR10Camera
        GAZEBO = GazeboGimballedCamera
        AUTO = make_gimballed_camera

    def __init__(
        self,
        camera_type: type[GimballedCamera],
        logs_path: Path,
        mapping_path: Optional[Path] = None,
        mapping_roi: Optional[np.ndarray] = None,
        pose_provider_override=None,
        enable_mapping=True,
        enable_benchmarking=False,
        enable_tracking=False,
        rosbag_testing=False,
        do_record=bool(os.getenv("DO_RECORD", True)),
    ):
        self._rosbag_testing = rosbag_testing
        self._enable_benchmarking = enable_benchmarking
        self._enable_tracking = enable_tracking
        self._do_record = do_record
        self._frames_since_autofocus = 0
        logs_path.mkdir(parents=True, exist_ok=True)

        self._logger = create_console_logger(logs_path, "perception")
        git_branch, git_commit = get_git_info()
        self._logger.info(f"Running on Git branch: {git_branch}, commit: {git_commit}")
        # get git diff and log to file
        with open(logs_path / "git_diff.txt", "w") as f:
            f.write(get_git_diff())

        self._camera = camera_type(logs_path / "primary")
        # self._backup_camera = (
        #     USBCam(logs_path / "secondary", relative_pose=USBCam.USBCamPose.DOWN.value)
        #     if camera_source == SiyiCam.Source.A8
        #     else GazeboCamera(
        #         logs_path / "secondary",
        #         "/world/map/model/iris/link/cam_down_link/sensor/camera/image",
        #         USBCam.USBCamPose.DOWN.value,
        #     )
        # )
        self._backup_camera = None
        # TODO: replace Pose.identity() here with actual offset of camera on the drone frame
        self._tracker = TargetTracker(
            track_logs_path=logs_path / "tracking",
            detection_logs_path=logs_path / "detection",
        )
        self._gimbal_flip_state: Perception.GimbalFlipState | None = None
        self._mode = Perception.Mode.IDLE
        self._drone_pose_provider = (
            DronePoseProvider(self._logger, logs_path / "drone_pose")
            if pose_provider_override is None
            else pose_provider_override
        )

        if mapping_roi is None:
            self._logger.warning("Using hard-coded mapping ROI")
            mapping_roi = np.array([[-100, -100, 0], [100, -100, 0], [-100, 100, 0]])

        if mapping_path is not None and mapping_roi is not None:
            self._mapper = DumbMapperSystem(self._logger, mapping_path, mapping_roi)
        else:
            self._mapper = None
        self._enable_mapping = enable_mapping

        self._loop_thread = threading.Thread(target=self.loop, daemon=True)
        self.looping = True

        self._tracking_cameras = (
            [
                GazeboCamera(
                    logs_path / "front_cam",
                    "/world/map/model/iris/link/avoidance_cam_front_link/sensor/camera/image",
                )
            ]
            if self._enable_tracking
            else []
        )

        self._drone_tracker = (
            ObstacleTracker(
                self._tracking_cameras,
                YoloObstacleDetector(),
                ParticleFilter(
                    region_size=torch.tensor([120, 120, 40], dtype=torch.float16)
                ),
            )
            if self._enable_tracking
            else None
        )
        # init target ID for gimbal lock
        self.current_target_id = None

        if self._enable_benchmarking:
            self.mem_logger = ProcessMemLogger()

        self._loop_thread.start()

    def get_drone_track_debug_img(self) -> np.ndarray | None:
        if self._drone_tracker is None:
            self._logger.warning(
                "Trying to get tracker debug image while we aren't running drone tracking"
            )
            return None
        return self._drone_tracker.get_debug_img()

    def get_camera(self):
        return self._camera

    def loop(self):
        while self.looping:
            try:
                self.loop_iter()
                if self._enable_benchmarking:
                    self._logger.debug("TICK")
                    self._logger.debug(self.mem_logger())
            except Exception as e:
                self._logger.exception(e)
                sleep(1)

    def get_obstacles(self):
        if self._drone_tracker is None:
            self._logger.warning(
                "Trying to get drone tracking obstacles while we aren't running drone tracking"
            )
            return []
        return self._drone_tracker.get_estimates()

    @profile
    def loop_iter(self):
        if self._frames_since_autofocus > 30:
            self._camera.do_autofocus()
            if self._gimbal_flip_state == Perception.GimbalFlipState.LOOKING_DOWN:
                self._camera.point_down()
            self._frames_since_autofocus = 0
        self._frames_since_autofocus += 1

        try:
            drone_pose = self._drone_pose_provider.get_pose_at_time(
                time(), self._rosbag_testing
            )
        except BufferError:
            return
        if drone_pose is None:
            self._logger.warning("No drone pose available")
            sleep(1)
            return
        if self._drone_tracker is not None:
            self._drone_tracker.update(drone_pose)
        if self._camera.recording:
            img = self._camera.get_latest_image()
        else:
            img = self._camera.take_image()

        # TODO: un-comment this when we get calibration params for the backup cam.
        # img_backup = None
        if self._backup_camera is not None:
            self._backup_camera.get_latest_image() if self._backup_camera.recording else self._backup_camera.take_image()

        if self._mode == Perception.Mode.IDLE:
            if self._camera.get_zoom_level() != 1.0:
                self._camera.set_absolute_zoom(1.0)
            GIMBAL_FLIP_ALTITUDE_METERS = 10
            if drone_pose.position[2] > GIMBAL_FLIP_ALTITUDE_METERS and (
                self._gimbal_flip_state is None
                or self._gimbal_flip_state == Perception.GimbalFlipState.LOOKING_FORWARD
            ):
                self._logger.info("Pointing camera down")
                self._camera.point_down()
                self._gimbal_flip_state = Perception.GimbalFlipState.LOOKING_DOWN
                if self._do_record and not self._camera.recording:
                    self.start_recording()
            elif drone_pose.position[2] <= GIMBAL_FLIP_ALTITUDE_METERS and (
                self._gimbal_flip_state is None
                or self._gimbal_flip_state == Perception.GimbalFlipState.LOOKING_DOWN
            ):
                self._logger.info("Pointing camera forward")
                self._camera.point_center()
                if self._mapper is not None:
                    self._mapper.save_map()
                self._gimbal_flip_state = Perception.GimbalFlipState.LOOKING_FORWARD

        if self._mode in (Perception.Mode.AREA_SCAN, Perception.Mode.TARGET_LOCK):
            if img is not None:
                cam_pose = self._camera.get_world_pose(drone_pose)
                self._tracker.update(img, cam_pose)
                tracks = self._tracker.get_tracks()
                self._logger.info(f"{len(tracks)} tracks")
                for t in self._tracker.get_tracks():
                    self._logger.debug(t)

        if self._mode == Perception.Mode.TARGET_LOCK:
            if self.current_target_id:
                track = self.get_target_positions()[self.current_target_id]
                target_pos_enu = np.array(
                    [track.position[0], track.position[1], track.position[2]]
                )
                self._logger.info(
                    f"Updated target lock to x: {track.position[0]}, y : {track.position[1]}"
                )
            else:
                target_pos_enu = np.array([0, 0, 0])
            drone_pos_enu = drone_pose.position
            drone_rot = drone_pose.rotation
            self._logger.info(f"drone rotation: {drone_rot}")
            vector_to_target = target_pos_enu - drone_pos_enu
            # Rotate the vector to the target into the drone's body frame
            drone_frame_vec = drone_rot.inv().apply([vector_to_target])
            # Calculate the yaw needed to point the camera at the target
            yaw = np.rad2deg(np.arctan2(drone_frame_vec[0][1], drone_frame_vec[0][0]))
            # don't account for the drone's pitch
            if self._camera == A8Camera:  # camera is from IRL
                pitch = np.rad2deg(
                    np.arctan2(
                        vector_to_target[0][2], np.linalg.norm(vector_to_target[0][0:2])
                    )
                )
            else:
                pitch = np.rad2deg(
                    np.arctan2(
                        drone_frame_vec[0][2], np.linalg.norm(drone_frame_vec[0][0:2])
                    )
                )
            # Set the camera attitude
            self._camera.set_absolute_position(yaw, pitch)

        if img is not None and self._enable_mapping and self._mapper is not None:
            if self._mapper.stitch_with_optimization(
                img, self._camera.get_world_pose(drone_pose)
            ):
                self._logger.debug("Stitching image into map")

    def set_area_scan(self, roi_local_coords: np.ndarray, img=None):
        """
        Set the area scan mode with the given region of interest coordinates.
        """
        self._logger.info("Set Perception mode to area scan")
        self._enable_mapping = False
        self._camera.point_center()
        sleep(2)
        self._camera.point_down()
        sleep(2)
        self._enable_mapping = True
        self._mode = Perception.Mode.AREA_SCAN

    def set_target_lock(self, target_id: int):
        self._logger.info("Set Perception mode to target lock")
        self.current_target_id = target_id
        self._enable_mapping = False
        self._camera.point_center()
        sleep(2)
        self._mode = Perception.Mode.TARGET_LOCK

    def set_idle(self):
        self._logger.info("Set Perception mode to idle")
        self._mode = Perception.Mode.IDLE
        self._camera.point_center()
        sleep(2)
        self._camera.point_down()
        sleep(2)
        self._enable_mapping = True

    def get_target_positions(self) -> list[TargetTracker.Track]:
        """
        Get the estimated positions of the targets in the region of interest.
        """
        return self._tracker.get_tracks()

    def start_recording(self):
        self._logger.info("Starting recording")
        self._camera.start_recording()
        if self._backup_camera is not None:
            self._backup_camera.start_recording()

    def stop_recording(self):
        self._logger.info("Stopping recording")
        self._camera.stop_recording()
        if self._backup_camera is not None:
            self._backup_camera.stop_recording()

    def get_mapping_coverage(self):
        if self._mapper is None:
            self._logger.warning("Mapping is not enabled")
            return np.zeros((1000, 1000), dtype=np.uint16)
        return self._mapper.get_coverage_image()

    def cleanup(self):
        self.looping = False
        self._loop_thread.join()
        self._mapper.save_map()
        del self._mapper
