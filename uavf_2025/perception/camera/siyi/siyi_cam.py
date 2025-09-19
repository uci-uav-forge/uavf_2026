from pathlib import Path

import numpy as np

from perception.types import Image, CameraPose
from scipy.spatial.transform import Rotation

from .siyi_sdk import SIYISDK
import subprocess
from shared.types import Pose

from ..gimballed_camera import GimballedCamera
from ..arducam_libraries.JetsonCamera import GstCamera


class SiyiCam(GimballedCamera):
    def __init__(
        self, log_dir: str | Path | None = None, cam_ip: str = "192.168.144.25"
    ):
        """
        Currently starts recording and logging as soon as constructed.
        This should be changed to after takeoff.
        """
        super().__init__(log_dir)
        stream_link = f"rtsp://{cam_ip}:8554/main.264"
        result = subprocess.run(
            ["gst-inspect-1.0", "nvvidconv"],
            env={"PAGER": "cat"},  # gst-inspect opens a pager without this
            capture_output=True,
            text=True,
        )

        # Check if the return code is zero and print the output
        video_convert_plugin = "nvvidconv" if result.returncode == 0 else "videoconvert"
        gst_pipeline = (
            f"rtspsrc location={stream_link} latency=10 ! "
            "rtpjitterbuffer latency=10 ! "
            "decodebin ! "
            f"{video_convert_plugin} ! "
            "video/x-raw, format=BGRx ! "
            "appsink drop=true"
        )

        self.cam = GstCamera(gst_pipeline)
        self.gimbal_control = SIYISDK(cam_ip)
        self.gimbal_control.connect()

    def take_image(self) -> Image[np.ndarray] | None:
        return Image(self.cam.getFrame()[:, :, :3])

    def set_absolute_position(self, yaw: float, pitch: float):
        if not self.gimbal_control.setAbsolutePosition(yaw, pitch):
            raise RuntimeError(
                "Failed to set absolute position: SIYI SDK returned false"
            )

    def set_gimbal_speed(self, yaw_speed: int, pitch_speed: int):
        if not self.gimbal_control.requestGimbalSpeed(yaw_speed, pitch_speed):
            raise RuntimeError("Failed to set gimbal speed: SIYI SDK returned false")

    def do_autofocus(self):
        if not self.gimbal_control.doAutoFocus():
            raise RuntimeError("Failed to autofocus: SIYI SDK returned false")

    def set_absolute_zoom(self, zoom_level: float):
        if not self.gimbal_control.setAbsoluteZoom(zoom_level):
            raise RuntimeError("Failed to set zoom level: SIYI SDK returned fase")

    def get_attitude(self):
        return self.gimbal_control.getAttitude()

    def get_attitude_speed(self):
        return self.gimbal_control.getAttitudeSpeed()

    def get_focal_length_px(self):
        raise NotImplementedError(
            "This is only implemented in subclasses because we hard-code two different implementations"
        )

    def get_zoom_level(self) -> float:
        return self.gimbal_control.getZoomLevel()

    @staticmethod
    def combine_drone_rot(cam_rot: Rotation, drone_rot: Rotation) -> Rotation:
        """
        The pitch and roll of the SIYI cameras is relative to the world frame, not the drone body frame.
        Therefore, we override this method to only use the drone's yaw to bring the yaw reported by the SIYI cam
        into the world frame.
        """
        drone_yaw = drone_rot.as_euler("zyx", degrees=True)[0]
        cam_yaw = cam_rot.as_euler(GimballedCamera.EULER_ORDER, degrees=True)[2]
        rotation = Rotation.from_euler("z", cam_yaw + drone_yaw, degrees=True) * cam_rot
        return rotation

    def get_world_pose(self, drone_pose: Pose) -> CameraPose:
        """
        Composes the drone pose with the relative pose of the camera to the drone.
        """
        attitude = self.get_attitude()
        position = drone_pose.position + drone_pose.rotation.apply(
            self.get_metadata().relative_pose.position
        )
        # the camera attitude roll and pitch are independent of the drone. They're already in world frame
        cam_rot = Rotation.from_euler(
            GimballedCamera.EULER_ORDER,
            [-attitude[1], attitude[2], attitude[0]],
            degrees=True,
        )

        rotation = SiyiCam.combine_drone_rot(cam_rot, drone_pose.rotation)

        ret = CameraPose(position, rotation, self.get_focal_length_px())

        self._pose_id += 1

        return ret
