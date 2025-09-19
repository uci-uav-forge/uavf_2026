import cv2
from tqdm import tqdm
from pathlib import Path
import json
from bisect import bisect_left
from shared.types import Pose
from perception.types import CameraPose, Image
import numpy as np
from scipy.spatial.transform import Rotation

from perception.odlc.target_tracking import TargetTracker
from perception.lib.osd import draw_tracks_on_img
from line_profiler import profile


class ODLCLogReplayer:
    """
    Replays our ODLC detection and tracking algos on a log directory
    containing drone pose data and images in the path `logs_dir/cam_dirname`.
    """

    def __init__(self, logs_dir: Path, cam_dirname):
        self.cam_dirname = cam_dirname

        self.logs_dir = logs_dir
        self.frame_stem_names = sorted(
            [fname.stem for fname in (self.logs_dir / self.cam_dirname).glob("*.jpg")],
            key=float,
        )

        if (logs_dir / "drone_pose").exists():
            self.drone_pose_data = [
                (
                    Pose.from_dict(
                        json.load(open(logs_dir / "drone_pose" / sorted_fname, "r"))
                    ),
                    float(sorted_fname.stem),
                )
                for sorted_fname in sorted(
                    (logs_dir / "drone_pose").glob("*.json"),
                    key=lambda x: float(x.stem),
                )
            ]
        else:
            raise RuntimeError("No drone pose data found in logs directory.")

        self._tracker = TargetTracker()
        cv2.namedWindow("map", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("map", 1080, 1080)

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

        return new_pose

    @profile
    def make_video(self, start_frame_index: int):
        # Concatenate the points along the x, y, and z directions
        for frame_fname in tqdm(self.frame_stem_names[start_frame_index::2]):
            frame = cv2.imread(f"{self.logs_dir}/{self.cam_dirname}/{frame_fname}.jpg")
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
                    f"{self.logs_dir}/{self.cam_dirname}/{frame_fname}.json", "r"
                ) as f:
                    data = json.loads(f.read()[:-1])
            except FileNotFoundError:
                data = dict()

            data["filename"] = f"{frame_fname}.jpg"

            if self.drone_pose_data is not None and "timestamp" in data:
                timestamp = data["timestamp"]
                drone_pose = self.interpolate_pose(timestamp)
                cam_position = np.array(data["relative_pose"]["position"])
                rot_quat = data["relative_pose"]["rotation_quat"]
                cam_rotation = Rotation.from_quat(rot_quat)
                # invert this:
                # Rotation.from_euler(
                #     "YXZ", [-attitude[1], attitude[2], attitude[0]], degrees=True
                # ),
                a = cam_rotation.as_euler("YXZ", degrees=True)
                attitude = [a[2], -a[0], a[1]]

                drone_tilt_angle = np.rad2deg(
                    np.arccos(np.dot(drone_pose.rotation.apply([0, 0, 1]), [0, 0, 1]))
                )
                if drone_tilt_angle > 30:
                    continue

                position = drone_pose.position + drone_pose.rotation.apply(cam_position)
                # the camera attitude roll and pitch are independent of the drone. They're already in world frame
                cam_rot = Rotation.from_euler(
                    "YX", [-attitude[1], attitude[2]], degrees=True
                )

                drone_yaw = drone_pose.rotation.as_euler("zyx", degrees=True)[0]
                cam_yaw = attitude[0]
                rotation = (
                    Rotation.from_euler("z", cam_yaw + drone_yaw, degrees=True)
                    * cam_rot
                )

                combined_pose = CameraPose(position, rotation, data["focal_len_px"])

                detections = self._tracker._detector.detect(Image(frame))
                self._tracker.update_with_detections(
                    Image(frame), detections, combined_pose
                )
                tracks = self._tracker.get_tracks()
                for d in detections:
                    bbox_xyxy = list(
                        map(
                            int,
                            [
                                d.bbox.x - d.bbox.width // 2,
                                d.bbox.y - d.bbox.height // 2,
                                d.bbox.x + d.bbox.width // 2,
                                d.bbox.y + d.bbox.height // 2,
                            ],
                        )
                    )
                    print("det id", d.det_id)
                    cv2.rectangle(
                        frame,
                        (bbox_xyxy[0], bbox_xyxy[1]),
                        (bbox_xyxy[2], bbox_xyxy[3]),
                        (255, 0, 0),
                        2,
                    )
                    cv2.circle(
                        frame, (int(d.bbox.x), int(d.bbox.y)), 5, (255, 0, 0), -1
                    )
                draw_tracks_on_img(frame, tracks, combined_pose)
                print("tracks len", len(tracks))
                cv2.imshow("map", frame)
                cv2.waitKey(0)


if __name__ == "__main__":
    log_dir = Path(
        "/home/forge/uavf_2025/logs/nvme/2025-06-12_14-22"
    )  # important points are frames 170*5 and (17+360)*5
    print(f"Processing log directory {log_dir}")
    cam_dirname = "primary" if (log_dir / "primary").exists() else "."

    lvm = ODLCLogReplayer(Path(log_dir), cam_dirname)
    lvm.make_video(start_frame_index=(130) * 5)
    print("Done")
    print(f"{log_dir}/video.mp4")
