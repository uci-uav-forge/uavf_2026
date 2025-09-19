import cv2 as cv
from itertools import chain
import traceback
from tqdm import tqdm
from pathlib import Path
import json
from bisect import bisect_left
from shared.types import Pose
from perception.lib.osd import draw_osd, draw_tracks_on_img
from perception.types import CameraPose
from perception.odlc.target_tracking import TargetTracker
from perception.camera.siyi import SiyiCam
from perception.camera.camera import Camera
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
from line_profiler import profile

from perception.odlc.target_detection import TargetDetector


class LogVideoMaker:
    def __init__(
        self,
        logs_dir: Path,
        cam_dirname: str,
        use_target_detector: bool = False,
        is_from_sim: bool = bool(os.getenv("SIM", False)),
        do_draw_osd=True,
    ):
        self.cam_dirname = cam_dirname
        self.is_from_sim = is_from_sim
        self.do_draw_osd = do_draw_osd

        self.yolo_model = None
        if use_target_detector:
            targetDector = TargetDetector()
            self.yolo_model = targetDector.model

        self.logs_dir = logs_dir
        self.frame_stem_names = sorted(
            [fname.stem for fname in (self.logs_dir / self.cam_dirname).glob("*.jpg")],
            key=float,
        )
        self.secondary_frame_stem_names = None
        if (self.logs_dir / "secondary").exists():
            self.secondary_frame_stem_names = sorted(
                [fname.stem for fname in (self.logs_dir / "secondary").glob("*.jpg")],
                key=float,
            )

            self.secondary_timestamps = []
            for frame_fname in self.secondary_frame_stem_names:
                try:
                    time = json.load(
                        open(
                            f"{self.logs_dir}/secondary/{frame_fname}.json",
                            "r",
                        )
                    )["timestamp"]
                    self.secondary_timestamps.append(time)
                except FileNotFoundError:
                    print(f"No JSON found for {frame_fname}")
                    pass
                except json.decoder.JSONDecodeError:
                    print(f"Corrupt JSON found for {frame_fname}")

            # TODO: Bc we are looping on top of images, we are not logging when json exists but no corresponding img.
            # Fix would be prob zipping and diffing

        first_frame = cv.imread(
            f"{logs_dir}/{self.cam_dirname}/{self.frame_stem_names[0]}.jpg"
        )
        double_first = np.vstack((first_frame, first_frame))
        first_timestamp = json.load(
            open(f"{logs_dir}/{self.cam_dirname}/{self.frame_stem_names[0]}.json", "r")
        )["timestamp"]
        second_timestamp = json.load(
            open(f"{logs_dir}/{self.cam_dirname}/{self.frame_stem_names[1]}.json", "r")
        )["timestamp"]
        self.video = cv.VideoWriter(
            f"{logs_dir}/video{'' if self.do_draw_osd else '_raw'}.mp4",
            cv.VideoWriter_fourcc(*"mp4v"),
            max(10, int(1 / (second_timestamp - first_timestamp))),
            double_first.shape[:2][::-1]
            if self.secondary_frame_stem_names is not None
            else first_frame.shape[:2][::-1],
        )

        if (logs_dir / "drone_pose").exists():
            self.drone_pose_data = []
            for sorted_fname in sorted(
                (logs_dir / "drone_pose").glob("*.json"),
                key=lambda x: float(x.stem),
            ):
                try:
                    self.drone_pose_data.append(
                        (
                            Pose.from_dict(
                                json.load(
                                    open(logs_dir / "drone_pose" / sorted_fname, "r")
                                )
                            ),
                            float(sorted_fname.stem),
                        )
                    )
                except json.decoder.JSONDecodeError:
                    print(f"Corrupt JSON found for {sorted_fname.stem}")
                    continue

        else:
            self.drone_pose_data = None

        if (logs_dir / "tracking").exists():
            self.tracking_logs_files = [
                (
                    sorted_fname,
                    float(sorted_fname.stem),
                )
                for sorted_fname in sorted(
                    (logs_dir / "tracking").glob("*.json"),
                    key=lambda x: float(x.stem),
                )
            ]
        else:
            self.tracking_logs_files = None

        if (logs_dir / "detection").exists():
            self.detection_data = []
            for sorted_fname in sorted(
                (logs_dir / "detection").glob("*.json"),
                key=lambda x: float(x.stem),
            ):
                try:
                    self.detection_data.append(
                        (
                            json.load(open(logs_dir / "detection" / sorted_fname, "r")),
                            float(sorted_fname.stem),
                        )
                    )
                except json.decoder.JSONDecodeError:
                    print(f"Corrupt JSON found for {sorted_fname.stem}")
                    continue
        else:
            self.detection_data = None

    def interpolate_pose(self, timestamp) -> Pose:
        if self.drone_pose_data[-1][1] < timestamp:
            return self.drone_pose_data[-1][0]
        if self.drone_pose_data[0][1] > timestamp:
            return self.drone_pose_data[0][0]

        idx = bisect_left([d[1] for d in self.drone_pose_data], timestamp)

        before = self.drone_pose_data[idx - 1]
        after = self.drone_pose_data[idx]
        proportion = (timestamp - before[1]) / (after[1] - before[1])
        new_pose = before[0].interpolate(after[0], proportion)

        if not 0 <= proportion <= 1:
            print(
                f"Warning: Interpolated pose proportion {proportion} is out of bounds [0, 1]."
            )

        return new_pose

    @profile
    def make_video(self, display_points: bool = False):
        # Concatenate the points along the x, y, and z directions
        for i in tqdm(range(len(self.frame_stem_names))):
            frame_fname = self.frame_stem_names[i]
            try:
                frame = cv.imread(
                    f"{self.logs_dir}/{self.cam_dirname}/{frame_fname}.jpg"
                )
                if self.do_draw_osd:
                    try:
                        data = json.load(
                            open(
                                f"{self.logs_dir}/{self.cam_dirname}/{frame_fname}.json",
                                "r",
                            )
                        )
                    except json.decoder.JSONDecodeError:
                        # if extra data error, try cutting off last character
                        with open(
                            f"{self.logs_dir}/{self.cam_dirname}/{frame_fname}.json",
                            "r",
                        ) as f:
                            data = json.loads(f.read()[:-1])
                    except FileNotFoundError:
                        data = dict()

                    try:
                        next_timestamp = json.load(
                            open(
                                f"{self.logs_dir}/{self.cam_dirname}/{self.frame_stem_names[i+1]}.json",
                                "r",
                            )
                        )["timestamp"]
                    except (FileNotFoundError, json.decoder.JSONDecodeError):
                        next_timestamp = None

                    data["filename"] = f"{frame_fname}.jpg"

                    if self.yolo_model is not None:
                        yolo_inf_res_raw = self.yolo_model(frame, verbose=False)
                        yolo_res = yolo_inf_res_raw[0]
                        # If there are bboxes then replace the frame with an annotated one
                        if yolo_res.boxes and len(yolo_res.boxes) > 0:
                            frame = yolo_res.plot()

                    if self.drone_pose_data is not None and "timestamp" in data:
                        timestamp = data["timestamp"]
                        closest_pose = self.interpolate_pose(timestamp)
                        cam_position = np.array(data["relative_pose"]["position"])
                        rot_quat = data["relative_pose"]["rotation_quat"]
                        cam_rotation = R.from_quat(rot_quat)

                        combined_pose = CameraPose(
                            closest_pose.position
                            + closest_pose.rotation.apply(cam_position),
                            Camera.combine_drone_rot(
                                cam_rotation, closest_pose.rotation
                            )
                            if self.is_from_sim
                            else SiyiCam.combine_drone_rot(
                                cam_rotation, closest_pose.rotation
                            ),
                            data["focal_len_px"],
                        )
                        del data["relative_pose"]
                        data["drone_position"] = closest_pose.position.tolist()
                        data["drone_roll_pitch_yaw"] = closest_pose.rotation.as_euler(
                            "xyz", degrees=True
                        ).tolist()
                        data["cam_pitch_roll_yaw"] = cam_rotation.as_euler(
                            SiyiCam.EULER_ORDER, degrees=True
                        ).tolist()

                        if self.tracking_logs_files is not None:
                            idx = bisect_left(
                                [d[1] for d in self.tracking_logs_files], timestamp
                            )
                            if idx == 0:
                                tracks = []
                            else:
                                tracks = TargetTracker.Track.list_from_json(
                                    self.tracking_logs_files[idx - 1][0]
                                )
                            draw_tracks_on_img(frame, tracks, combined_pose)
                            data["targets detected"] = len(tracks)

                        if self.detection_data is not None:
                            idx = bisect_left(
                                [d[1] for d in self.detection_data], timestamp
                            )
                            if (
                                idx < len(self.detection_data)
                                and next_timestamp is not None
                                and next_timestamp > self.detection_data[idx][1]
                            ):
                                detections = self.detection_data[idx][0]
                                for d in detections:
                                    bbox_xywh = d["bbox"]
                                    cv.rectangle(
                                        frame,
                                        (
                                            int(bbox_xywh[0] - bbox_xywh[2] / 2),
                                            int(bbox_xywh[1] - bbox_xywh[3] / 2),
                                        ),
                                        (
                                            int(bbox_xywh[0] + bbox_xywh[2] / 2),
                                            int(bbox_xywh[1] + bbox_xywh[3] / 2),
                                        ),
                                        (0, 0, 255),
                                        2,
                                    )

                        draw_osd(
                            frame, data, combined_pose, display_points, closest_pose
                        )
                        # find closest secondary frame
                        if self.secondary_frame_stem_names is not None:
                            idx = bisect_left(self.secondary_timestamps, timestamp)
                            secondary_frame = cv.imread(
                                f"{self.logs_dir}/secondary/{self.secondary_frame_stem_names[idx]}.jpg"
                            )
                            secondary_data = json.load(
                                open(
                                    f"{self.logs_dir}/secondary/{self.secondary_frame_stem_names[idx]}.json",
                                    "r",
                                )
                            )

                            secondary_cam_position = np.array(
                                secondary_data["relative_pose"]["position"]
                            )
                            secondary_rot_quat = secondary_data["relative_pose"][
                                "rotation_quat"
                            ]
                            secondary_cam_rotation = R.from_quat(secondary_rot_quat)

                            secondary_combined_pose = CameraPose(
                                closest_pose.position
                                + closest_pose.rotation.apply(secondary_cam_position),
                                closest_pose.rotation * secondary_cam_rotation,
                                secondary_data["focal_len_px"],
                            )

                            del secondary_data[
                                "relative_pose"
                            ]  # remove relative pose from OSD
                            secondary_data["filename"] = (
                                f"{self.secondary_frame_stem_names[idx]}.jpg"
                            )
                            draw_osd(
                                secondary_frame,
                                secondary_data,
                                secondary_combined_pose,
                                display_points,
                                closest_pose,
                            )
                            combined_frame = np.vstack((frame, secondary_frame))
                            self.video.write(combined_frame)
                            continue

                self.video.write(frame)
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    print("Exiting due to keyboard interrupt")
                    break
                print(f"Error processing frame {frame_fname}")
                traceback.print_exc()

        self.video.release()


if __name__ == "__main__":
    # list all folders in the directory this file is in
    all_log_dirs = list(
        chain(
            Path(__file__).parent.glob("????-??-??_??-??"),
            Path(__file__).parent.glob("*/????-??-??_??-??"),
            Path(__file__).parent.glob("*/*/????-??-??_??-??"),
        )
    )  # assuming all log directories are named with the format YYYY-MM-DD_HH-MM
    log_dirs = [
        log_dir
        for log_dir in all_log_dirs
        if "usb" not in str(log_dir) and "old" not in str(log_dir)
    ]
    # print out all them sorted by name with indices
    for idx, log_dir in enumerate(sorted(log_dirs)):
        cam_dirname = "primary" if (log_dir / "primary").exists() else "."
        num_frames = sum(1 for _ in (log_dir / cam_dirname).glob("*.jpg"))
        print(f"{idx}\t{num_frames} frames\t{str(log_dir).split('logs/')[-1]}")
    # prompt user to enter index of the log directory they want to process (or no input to select most recent)
    log_dir_selected = input(
        "Enter index of log directory to process (default -1, last one in list): "
    )
    if log_dir_selected == "":
        log_dir_idx = -1
    else:
        log_dir_idx = int(log_dir_selected)
    display_points = False
    do_draw_osd = input("Draw osd? (y/n): ")
    if do_draw_osd == "y":
        do_draw_osd = True
    else:
        do_draw_osd = False
    log_dir = sorted(log_dirs)[log_dir_idx]
    print(f"Processing log directory {log_dir}")
    cam_dirname = "primary" if (log_dir / "primary").exists() else "."

    lvm = LogVideoMaker(
        Path(log_dir), cam_dirname, use_target_detector=False, do_draw_osd=do_draw_osd
    )
    lvm.make_video(display_points)
    print("Done")
    print(f"{log_dir}/video.mp4")
