import cv2 as cv
from tqdm import tqdm
from pathlib import Path
import json
from bisect import bisect_left
from shared.types import Pose
from perception.types import CameraPose, Image
import numpy as np
from scipy.spatial.transform import Rotation

from perception.mapping.dumb_mapper import DumbMapperSystem


class LogVideoMaker:
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
            self.drone_pose_data = None

        self.mapper = DumbMapperSystem(
            None,
            None,
            np.array(
                [
                    [-127.57107784994488, 138.83403588150827, 0.0],
                    [198.63264651566595, 69.6808019983323, 0.0],
                    [-155.821122249676, 7.962380827915501, 0.0],
                ]
            ),
        )
        cv.namedWindow("map", cv.WINDOW_NORMAL)
        cv.resizeWindow("map", 1080, 1080)

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

    def make_video(self, display_points: bool = False):
        # Concatenate the points along the x, y, and z directions
        for frame_fname in tqdm(self.frame_stem_names[::1]):
            frame = cv.imread(f"{self.logs_dir}/{self.cam_dirname}/{frame_fname}.jpg")
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
                rotation = drone_pose.rotation * cam_rotation

                combined_pose = CameraPose(position, rotation, data["focal_len_px"])

                self.mapper.stitch_with_optimization(
                    Image(frame[:, 1920 // 2 - 1080 // 2 : 1920 // 2 + 1080 // 2, :]),
                    combined_pose,
                )
                # self.mapper.stitch(Image(frame), combined_pose)
                cv.imshow("map", self.mapper.get_map())
                coverage = self.mapper.get_coverage_image()
                scaled = coverage / coverage.max() if coverage.max() > 0 else coverage
                cv.imshow("coverage", (scaled * 255).astype(np.uint8))
                cv.waitKey(0)
                # find closest secondary frame
        cv.waitKey(0)


if __name__ == "__main__":
    log_dir = Path("/home/forge/ardu_ws/src/uavf_2025/logs/0-2025-06-21/14-17")
    print(f"Processing log directory {log_dir}")
    cam_dirname = "primary" if (log_dir / "primary").exists() else "."

    lvm = LogVideoMaker(Path(log_dir), cam_dirname)
    lvm.make_video()
