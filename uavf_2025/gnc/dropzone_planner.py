from __future__ import annotations

import numpy as np
from perception.odlc.target_tracking import TargetTracker
from gnc.util import local_to_gps, validate_points
from shared.payload_manager import PayloadManager
from gnc.mapping_custom import get_mapping_path
from shapely.geometry import Polygon, LineString
from shapely.ops import split
from gnc.mapping_tsp import MappingPathPlanner

import math
from time import sleep
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Prevents circular import
    from gnc.commander_node import (
        CommanderNode,
    )
from .constants import (
    TARGET_ALTITUDE,
    SCAN_ALTITUDE,
    TURN_LENGTH,
    SCAN_SPEED,
    MAPPING_SPEED,
)


class DropzonePlanner:
    """
    Handles all logic related to controlling drone motion during the payload drop.
    """

    def __init__(self, commander: CommanderNode):
        self.commander = commander
        self.payload_client = PayloadManager(commander)
        self.targets_dropped: list[TargetTracker.Track] = []
        self.centroids = []

    def gen_dropzone_plan(self) -> list[tuple[float, float]]:
        """
        Generates dropzone plan given the drone current position and
        the dropzone coords. Gets these from the commander_node.
        """

        # Get dropzone bounds
        dropzone_coords = self.commander.dropzone_bounds_mlocal

        # Retrieve current x y coordinates
        cur_xy = self.commander.pose_provider.get_local_pose().position[:2]
        self.commander.log(f"bounds: {dropzone_coords}")
        self.commander.log(f"current xy: {cur_xy}")

        # Finds the closest corner of the dropzone
        closest_idx = min(
            range(4), key=lambda i: float(np.linalg.norm(dropzone_coords[i] - cur_xy))
        )
        closest_corner = dropzone_coords[closest_idx]

        # Finds the closest point to the corner (finding the closest short side of the rectangle)
        min_distance = float("inf")
        closest_to_corner = []
        all_corners_idx_except_closest = filter(lambda j: j != closest_idx, range(4))
        for i in all_corners_idx_except_closest:
            point = dropzone_coords[i]
            distance = math.dist(closest_corner, point)
            if distance < min_distance:  #  if closer than other corners
                min_distance = distance
                closest_to_corner = point

        if len(closest_corner) == 0:
            raise RuntimeError("Expected at least 2 dropzone corners for boundary.")

        # Finds the closest midpoint to the drone
        midpoint_close = (
            (closest_corner[0] + closest_to_corner[0]) / 2,
            (closest_corner[1] + closest_to_corner[1]) / 2,
        )

        self.commander.log(f"Closest midpoint is {midpoint_close}")
        self.commander.log(
            f"GPS: {local_to_gps(midpoint_close, self.commander.global_home)}"
        )

        # Finds the opposing two points
        opposite_points = []
        for point in dropzone_coords:
            if (
                math.dist(point, closest_corner) > 1e-3
                and math.dist(point, closest_to_corner) > 1e-3
            ):
                opposite_points.append(point)

        # Finds the furthest midpoint from the drone
        midpoint_far = (
            (opposite_points[0][0] + opposite_points[1][0]) / 2,
            (opposite_points[0][1] + opposite_points[1][1]) / 2,
        )
        self.commander.log(f"Furthest midpoint is {midpoint_far}")
        self.commander.log(
            f"GPS: {local_to_gps(midpoint_far, self.commander.global_home)}"
        )

        # Add first waypoint to allow for turning before entering dropzone
        dist = (
            (midpoint_close[0] - midpoint_far[0]) ** 2
            + (midpoint_close[1] - midpoint_far[1]) ** 2
        ) ** 0.5
        x = (
            midpoint_close[0]
            - TURN_LENGTH * (midpoint_far[0] - midpoint_close[0]) / dist
        )
        y = (
            midpoint_close[1]
            - TURN_LENGTH * (midpoint_far[1] - midpoint_close[1]) / dist
        )
        extender_point = (x, y)

        # Waypoint list
        result_wps = [extender_point, midpoint_close, midpoint_far]

        return result_wps

    def gen_dropzone_plan_gps(
        self,
    ) -> tuple[list[tuple[float, float, float]], np.ndarray]:
        """
        Converts the drop_zone plan waypoints to GPS
        """

        dropzone_plan = self.gen_dropzone_plan()

        # Logging the way points
        self.commander.log(f"Local coords: {dropzone_plan}")
        self.commander.log(
            f"Planned waypoints: {[local_to_gps(wp, self.commander.global_home) for wp in dropzone_plan]}"
        )

        dropzone_wps = []
        for point in dropzone_plan:
            dropzone_wps.append(
                (
                    *local_to_gps(point, self.commander.global_home),
                    SCAN_ALTITUDE,
                )
            )

        dropzone_coords = np.array(self.commander.get_dropzone_bounds_mlocal())

        # Executing the way points
        self.commander.log(f"Dropzone_plan: {dropzone_wps}")
        return dropzone_wps, dropzone_coords

    def on_dropzone_scan_entry(self, dropzone_coords):
        """Function to execute once the first dropzone waypoint reached."""
        self.commander.log("Dropzone scan entry func executing...")
        self.commander.change_speed(SCAN_SPEED)
        self.commander.perception.set_area_scan(dropzone_coords)

    def scan_dropzone(self) -> None:
        """
        Fly straight line at high altitude accross drop zone.
        """
        self.commander.change_speed(None)
        dropzone_wps, dropzone_coords = self.gen_dropzone_plan_gps()
        self.commander.execute_waypoints(
            dropzone_wps,
            after_second_waypoint_cb=lambda: self.on_dropzone_scan_entry(
                dropzone_coords
            ),
            set_yaw=True,
        )

        self.commander.perception.set_idle()

        self.commander.change_speed(None)

    def target_path(self) -> None:
        """
        Generates dropzone plan to fly to new targets.
        """

        # get target tracks from perception
        target_tracks = self.commander.filter_bad_targets(
            self.commander.perception.get_target_positions()
        )

        target_path_plan = self.filter_targets(target_tracks, 3)
        self.commander.log(f"The target path plan is {target_path_plan}.")
        self.commander.log_status(f"Target path: {target_path_plan}")

        if len(target_path_plan) == 0:
            self.drop_middle()

        for perc_index, track in target_path_plan:
            initial_prob = max(track.probs)

            # Update target position
            # need to pass in index
            self.commander.fly_to_target_local(
                track.position[0], track.position[1], TARGET_ALTITUDE, perc_index
            )

            # Check if probability fell over 50% and that there are still targets to drop on after this
            final_prob = max(track.probs)
            if final_prob / initial_prob < 0.5 and target_path_plan[-1][1] != track:
                self.commander.log(
                    f"Probability fell {100 * final_prob / initial_prob}%, trying another target..."
                )
                continue
            self.commander.log(
                f"Dropping on approved target => initial_prob: {initial_prob}, final_prob: {final_prob}"
            )
            self.commander.log_status("Beginning to drop...")

            sleep(3)  # Give 3 seconds to stabilize over target
            self.targets_dropped.append(track)
            self.commander.log("Dropping payload...")
            self.payload_client.drop_payload()  # Wait to finish the drop
            break  # can't drop more than once per lap

    def drop_middle(self):
        """Drop payload in the center of the dropzone if no targets detected."""
        self.commander.log("Executing polgyon-based split for drop")
        current_x, current_y, _ = self.commander.get_pose_position()
        if len(self.centroids) == 0:
            coords = self.commander.get_dropzone_bounds_mlocal()
            if not np.allclose(coords[0], coords[-1]):
                coords = list(coords) + [coords[0]]

            polygon = Polygon(coords)

            if not polygon.is_valid:
                raise ValueError("Dropzone polygon is not valid")

            min_x, min_y, max_x, max_y = polygon.bounds
            split_x = (min_x + max_x) / 2
            split_line = LineString([(split_x, min_y - 1), (split_x, max_y + 1)])

            result = split(polygon, split_line)

            if len(result.geoms) != 2:
                raise RuntimeError("Expected polygon to split into two")

            # Choose one half: Option A - closest to current drone location
            self.centroids = [(poly.centroid.coords[0], poly) for poly in result.geoms]
        centroid_pack = min(
            self.centroids, key=lambda tup: math.dist(tup[0], (current_x, current_y))
        )
        chosen_centroid, chosen_poly = centroid_pack
        x, y = chosen_centroid
        self.centroids.remove(centroid_pack)
        self.commander.log_status(f"Flying to single-drop centroid: ({x}, {y})")
        self.commander.fly_to_target_local(x, y, TARGET_ALTITUDE, index=-1)
        sleep(3)  # Allow time to stabilize
        self.commander.log("Dropping payload at selected polygon half")
        self.payload_client.drop_payload()

    def filter_targets(
        self, targets: list[TargetTracker.Track], desired_num: int
    ) -> list[tuple[int, TargetTracker.Track]]:
        """
        Given a list of Tracks, return the subset that holds the desired_num of
        Tracks to fly to.
        """
        qualified_targets = filter(lambda target: self.target_reqs(target), targets)
        targets_filtered = sorted(
            qualified_targets,
            key=lambda target: -len(target.contributing_detections) * max(target.probs),
        )[:desired_num]

        return self.generate_ordered_targets_list(targets_filtered)

    def generate_ordered_targets_list(
        self, targets: list[TargetTracker.Track]
    ) -> list[tuple[int, TargetTracker.Track]]:
        """
        Return the list of Tracks in ascending order of distance from current location.
        """
        current_x, current_y, _ = self.commander.get_pose_position()
        self.commander.log(
            f"Generating ordered targets list based on the drone's current position of ({current_x}, {current_y})."
        )

        target_to_distance_mapping = {}

        for t in targets:
            target_x, target_y = t.position[0], t.position[1]
            target_to_distance_mapping[t] = math.sqrt(
                (current_x - target_x) ** 2 + (current_y - target_y) ** 2
            )

        self.commander.log(f"Target to distance mapping: {target_to_distance_mapping}.")
        ordered_targets = sorted(
            targets, key=lambda target: target_to_distance_mapping[target]
        )

        target_tracks = self.commander.filter_bad_targets(
            self.commander.perception.get_target_positions()
        )

        indexed_targets = [
            (target_tracks.index(track), track) for track in ordered_targets
        ]

        return indexed_targets

    def mapping_path(self, height: float, speed: float, margin_of_error: float) -> None:
        """
        :params
        height: height in meters to fly at
        speed: speed in m/s to fly at
        margin_of_error: distance in meters of how close the drone needs to get to each
        coordinate on the mapping path before switching to the next coordinate
        Fly along a path to map the ROI.
        """
        # img = self.commander.perception.get_mapping_coverage()
        img = np.zeros((1000, 1000), dtype=np.uint8)
        current_pos = self.commander.get_pose_position()
        self.commander.log(
            f"Generating mapping path starting from current local position: {current_pos}"
        )
        path: list[tuple[float, float, float]] = get_mapping_path(
            img, current_pos, height, *self.commander.get_roi_corners_local()
        )
        if not path:
            self.commander.log("Empty mapping path. Aborting mapping.")
            return
        self.commander.log(f"Mapping path: {path}")
        validate_points(
            [local_to_gps(coord[0:2], self.commander.global_home) for coord in path],
            self.commander.geofence,
            False,
        )
        self.commander.log("Mapping path verified to be within geofence.")
        self.commander.execute_waypoints(path, mapping=True)

    def mapping_path_tsp(
        self, height: float, speed: float, margin_of_error: float
    ) -> None:
        """
        :params
        height: height in meters to fly at
        speed: speed in m/s to fly at
        margin_of_error: distance in meters of how close the drone needs to get to each
        coordinate on the mapping path before switching to the next coordinate
        Fly along a path to map the ROI.
        """
        heatmap = self.commander.perception.get_mapping_coverage()
        current_pos = self.commander.get_pose_position()
        self.commander.log(
            f"Generating mapping path starting from current local position: {current_pos}"
        )
        PathPlanner = MappingPathPlanner(
            heatmap, self.commander.get_roi_corners_local(), height, 1
        )
        path = PathPlanner.construct_path(current_pos)
        PathPlanner.save_path_image(heatmap)
        if not path:
            self.commander.log("Empty mapping path. Aborting mapping.")
            return
        self.commander.log(f"Mapping path: {path}")
        validate_points(
            [local_to_gps(coord[0:2], self.commander.global_home) for coord in path],
            self.commander.geofence,
            False,
        )
        self.commander.log("Mapping path verified to be within geofence.")
        self.commander.change_speed(MAPPING_SPEED)
        self.commander.execute_waypoints(path, mapping=True)

    def target_reqs(self, target) -> bool:
        """
        Returns true if a track is inside the dropzone and not already dropped on.
        """
        gps = [
            local_to_gps(
                (target.position[0], target.position[1]), self.commander.global_home
            )
        ]
        try:
            validate_points(gps, self.commander.dropzone_bounds, False)
        except AssertionError:
            self.commander.log(
                f"Target not inside dropzone, disqualifying target {target} at {gps}."
            )
            return False

        return target not in self.targets_dropped
