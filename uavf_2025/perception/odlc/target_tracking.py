from __future__ import annotations
from pathlib import Path
from time import time
import json

import numpy as np

from perception.types import Image, CameraPose, TargetBBox
from perception.odlc.target_detection import TargetDetector
from perception.odlc.localization import Localizer

TARGET_SIZE_THRESHOLD = 5  # largest targets will be >3 meter wingspan model planes


class TargetTracker:
    class Track:
        _track_id: int = 0

        def __init__(
            self, det: TargetBBox, det_pos: np.ndarray, track_id: int = -1
        ) -> None:
            # assign unique ids to distinct tracks
            if track_id == -1:
                self.id: int = TargetTracker.Track._track_id
                TargetTracker.Track._track_id += 1
            else:
                self.id = track_id
                TargetTracker.Track._track_id = max(
                    track_id, TargetTracker.Track._track_id
                )

            self.position: np.ndarray = det_pos
            self.probs: np.ndarray = det.cls_probs
            self.contributing_detections: list[tuple[int, int]] = [
                (det.img_id, det.det_id)
            ]
            """
            list of tuples (img_id, det_id)
            """
            self.det = det

            # to keep track of moving average in an efficient way
            self._cumulative_sum: np.ndarray = self.position
            self._num_detections: int = 1

        def __repr__(self):
            return f"{self.position}, {self._num_detections}, {self.id}"

        def add_detection(self, det: TargetBBox, det_pos: np.ndarray) -> None:
            # recalculate position
            self._cumulative_sum += det_pos
            self._num_detections += 1
            self.position = self._cumulative_sum / self._num_detections
            # update contributing detections
            self.contributing_detections.append((det.img_id, det.det_id))
            self.det = det

            # TODO: Take the new class probabilities into account via averaging or something

        @staticmethod
        def list_from_json(json_path: Path) -> list[TargetTracker.Track]:
            with open(json_path, "r") as f:
                tracks_json = json.load(f)
            tracks = []
            for track_json in tracks_json:
                track = TargetTracker.Track.__new__(TargetTracker.Track)
                track.id = track_json["id"]
                track.position = np.array(track_json["position"])
                track.probs = np.array(track_json["probs"])
                track.contributing_detections = track_json["contributing_detections"]
                tracks.append(track)
            return tracks

    def __init__(
        self,
        tolerance=5.0,
        track_logs_path: Path | None = None,
        detection_logs_path: Path | None = None,
    ):
        self._detector = TargetDetector()
        self._tracks: list[TargetTracker.Track] = []
        self._tolerance = tolerance

        # logging
        self._track_logs_path = track_logs_path
        if self._track_logs_path is not None:
            self._track_logs_path.mkdir(parents=True, exist_ok=True)
        self._track_log_id = 0

        self._detection_logs_path = detection_logs_path
        if self._detection_logs_path is not None:
            self._detection_logs_path.mkdir(parents=True, exist_ok=True)

    def update(self, image: Image, pose: CameraPose) -> None:
        detections = self._detector.detect(image)
        return self.update_with_detections(image, detections, pose)

    def update_with_detections(self, image, detections, pose):
        localizer = Localizer.from_focal_length(
            (image.shape[1], image.shape[0]),
            Localizer.drone_initial_directions(),
            ground_axis=2,
            ground_coordinate=0,
        )
        for det in detections:
            # get position of detection
            # det_xyxy_3d = [
            #     localizer.coords_2d_to_3d(x, y, pose)
            #     for x, y in [
            #         (det.bbox.x - det.bbox.width / 2, det.bbox.y - det.bbox.height / 2),
            #         (det.bbox.x + det.bbox.width / 2, det.bbox.y - det.bbox.height / 2),
            #         (det.bbox.x + det.bbox.width / 2, det.bbox.y + det.bbox.height / 2),
            #         (det.bbox.x - det.bbox.width / 2, det.bbox.y + det.bbox.height / 2),
            #     ]
            # ]
            # diagonal_length = np.max(
            #     [
            #         np.linalg.norm(det_xyxy_3d[2] - det_xyxy_3d[0]),
            #         np.linalg.norm(det_xyxy_3d[1] - det_xyxy_3d[3]),
            #     ]
            # )
            # if diagonal_length > TARGET_SIZE_THRESHOLD:
            #    print("Culling target")
            #    continue

            det_pos = localizer.prediction_to_coords(det, pose)
            det_pos = np.array([det_pos.x, det_pos.y, det_pos.z])

            # intialize tracks if there are none yet
            if not self._tracks:
                self._tracks.append(TargetTracker.Track(det, det_pos))
                continue

            has_been_added = False
            for tr in self._tracks:
                break
                if det.det_id in {
                    det_id for img_id, det_id in tr.contributing_detections
                }:
                    tr.add_detection(det, det_pos)
                    has_been_added = True
                    break
            if not has_been_added:
                # find nearest track
                nearest_track: TargetTracker.Track = min(
                    self._tracks, key=lambda tr: np.linalg.norm(det_pos - tr.position)
                )

                # determine if nearest track is "close enough" and is the same object as current detection
                close_enough = (
                    np.linalg.norm(nearest_track.position - det_pos) < self._tolerance
                )

                if close_enough:
                    # update nearest track
                    nearest_track.add_detection(det, det_pos)
                else:
                    # create new track
                    self._tracks.append(TargetTracker.Track(det, det_pos, det.det_id))

        self._log_tracks()
        self._log_detections(detections)

    def _log_tracks(self) -> None:
        if self._track_logs_path is None:
            return
        json.dump(
            [
                {
                    "id": track.id,
                    "position": track.position.tolist(),
                    "probs": list(track.probs),
                    "contributing_detections": track.contributing_detections,
                }
                for track in self._tracks
            ],
            open(self._track_logs_path / f"{time()}.json", "w"),
        )
        self._track_log_id += 1

    def _log_detections(self, detections: list[TargetBBox]) -> None:
        if self._detection_logs_path is None:
            return
        json.dump(
            [
                {
                    "bbox": [
                        int(detection.bbox.x),
                        int(detection.bbox.y),
                        int(detection.bbox.width),
                        int(detection.bbox.height),
                    ],
                    "cls_probs": list(detection.cls_probs),
                    "img_id": detection.img_id,
                    "det_id": detection.det_id,
                }
                for detection in detections
            ],
            open(self._detection_logs_path / f"{time()}.json", "w"),
        )

    def get_tracks(self) -> list[Track]:
        return self._tracks
