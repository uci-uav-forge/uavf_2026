# Run this file AFTER the sim has started and it will randomly spawn targets every 10 seconds

import rclpy
import rclpy.qos
from rclpy.node import Node
from launch import LaunchDescription, LaunchService
from launch_ros.actions import Node as LaunchNode
from mavros_msgs.msg import HomePosition
from std_srvs.srv import Empty
import random
import sys
import os
from gnc.util import read_gpx_file
from pymap3d import geodetic2enu
import numpy as np
import rclpy.subscription
import time
from pathlib import Path
import subprocess
import threading
import json


def suppress_stdout():
    sys.stdout = open(os.devnull, "w")
    # sys.stderr = open(os.devnull, 'w')


def spawn_model_launch_description(model_path, model_name, x, y, z):
    return LaunchDescription(
        [
            LaunchNode(
                package="ros_gz_sim",
                executable="create",
                name="spawn_" + model_name,
                output="screen",
                arguments=[
                    "-file",
                    model_path,
                    "-name",
                    model_name,
                    "-x",
                    str(x),
                    "-y",
                    str(y),
                    "-z",
                    str(z),
                    "--ros-args",
                    "--log-level",
                    "error",
                ],
            )
        ]
    )


def delete_model(model_name: str):
    # manual command: gz service -s /world/map/remove --reqtype gz.msgs.Entity --req 'type: MODEL; name: "target_0"' --reptype gz.msgs.Boolean
    subprocess.run(
        [
            "gz",
            "service",
            "-s",
            "/world/map/remove",
            "--reqtype",
            "gz.msgs.Entity",
            "--req",
            f'type: MODEL; name: "{model_name}"',
            "--reptype",
            "gz.msgs.Boolean",
        ]
    )


TARGET_RADIUS = 5  # 7.62 is 25 feet in meters, but this is reduced because some of our dropzone definitions are too small.

ARDU_WS_SRC_PATH = str(Path.home() / "ardu_ws/src")


def random_point_in_pca_rect(points):
    """
    Finds the best-fit oriented rectangle using PCA and samples a random point within it.

    Args:
        points (np.ndarray): Nx2 array of (x, y) coordinates.

    Returns:
        np.ndarray: A single (x, y) point sampled uniformly within the rectangle.
    """
    points = np.asarray(points)
    centroid = np.mean(points, axis=0)

    # Compute PCA using SVD
    U, S, Vt = np.linalg.svd(points - centroid)
    axes = Vt.T  # Principal component directions

    # Project points onto PCA axes
    proj_points = (points - centroid) @ axes
    min_proj, max_proj = proj_points.min(axis=0), proj_points.max(axis=0)

    # Sample a random point in the aligned bounding box
    rand_proj = np.random.uniform(min_proj, max_proj)

    # Transform back to original space
    return centroid + rand_proj @ axes.T


class GzPositionProvider:
    def __init__(self, topic: str = "/world/map/model/iris/joint_state"):
        self.topic = topic
        self.position = np.zeros(3)  # Initialize as [0.0, 0.0, 0.0]
        self.quaternion = np.array(
            [0.0, 0.0, 0.0, 1.0]
        )  # Initialize as [0.0, 0.0, 0.0, 1.0]
        self.process = None
        self.thread = None
        self.running = True
        self.thread = threading.Thread(target=self._run_process, daemon=True)
        self.thread.start()

    def _parse_json_output(self, json_line):
        """
        Parses a JSON string for position and orientation.
        Updates the position and quaternion variables.
        """
        try:
            data = json.loads(json_line)
            pose = data.get("pose", {})
            position = pose.get("position", {})
            orientation = pose.get("orientation", {})

            # Update instance variables with NumPy arrays
            self.position = np.array(
                [position.get("x", 0.0), position.get("y", 0.0), position.get("z", 0.0)]
            )
            self.quaternion = np.array(
                [
                    orientation.get("x", 0.0),
                    orientation.get("y", 0.0),
                    orientation.get("z", 0.0),
                    orientation.get("w", 1.0),
                ]
            )
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

    def _run_process(self):
        """
        Runs the gz topic command and continuously reads the output.
        """
        command = ["gz", "topic", "--echo", "--json-output", "--topic", self.topic]
        self.process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        while self.running:
            line = self.process.stdout.readline()
            if line.strip():
                self._parse_json_output(line)

        self.process.terminate()

    def __del__(self):
        """
        Stops the subprocess and processing thread.
        """
        if self.running:
            self.running = False
            self.thread.join()

    def get_position(self):
        return self.position


class DynamicSpawner(Node):
    def __init__(
        self,
        gpx_filepath: str = str(
            Path(__file__).parent.parent.parent / "gnc/gcs/data/suas_runway_1.gpx"
        ),
    ):
        super().__init__("dynamic_spawner")

        # Add target model paths here
        self.target_models = [
            f"{ARDU_WS_SRC_PATH}/ardupilot_gazebo/models/stop_sign",
            f"{ARDU_WS_SRC_PATH}/ardupilot_gazebo/models/person_standing",
            f"{ARDU_WS_SRC_PATH}/ardupilot_gazebo/models/prius_hybrid",
            f"{ARDU_WS_SRC_PATH}/ardupilot_gazebo/models/robocup_3Dsim_ball",
            f"{ARDU_WS_SRC_PATH}/ardupilot_gazebo/models/motorcycle_0",
            f"{ARDU_WS_SRC_PATH}/ardupilot_gazebo/models/boat_0",
            f"{ARDU_WS_SRC_PATH}/ardupilot_gazebo/models/bat_0",
            f"{ARDU_WS_SRC_PATH}/ardupilot_gazebo/models/bed_0",
            f"{ARDU_WS_SRC_PATH}/ardupilot_gazebo/models/plane_0",
            f"{ARDU_WS_SRC_PATH}/ardupilot_gazebo/models/bus_0",
            f"{ARDU_WS_SRC_PATH}/ardupilot_gazebo/models/skis_0",
            f"{ARDU_WS_SRC_PATH}/ardupilot_gazebo/models/snowboard_0",
            f"{ARDU_WS_SRC_PATH}/ardupilot_gazebo/models/suitcase_0",
            f"{ARDU_WS_SRC_PATH}/ardupilot_gazebo/models/tennis_racket_0",
            f"{ARDU_WS_SRC_PATH}/ardupilot_gazebo/models/umbrella_0",
        ]
        self.beacon_model_path = f"{ARDU_WS_SRC_PATH}/ardupilot_gazebo/models/beacon"

        random.shuffle(self.target_models)

        self.home_pos_geo = None
        self.home_pos_subscriber = self.create_subscription(
            HomePosition,
            "/mavros/home_position/home",
            self._home_pos_listener_callback,
            qos_profile=rclpy.qos.QoSProfile(
                depth=1, reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT
            ),
        )

        self.drop_service = self.create_service(
            Empty, "uavf_drop_srv", self._drop_callback
        )

        self.spawn_count = 0

        self.gpx_tracks = read_gpx_file(gpx_filepath)
        self.target_positions_enu = []
        self.hit_counts = np.zeros(4)
        self.drops_done = 0
        self.run_name = time.strftime("%H-%M-%S")
        self.position_provider = GzPositionProvider()

    def _home_pos_listener_callback(self, msg: HomePosition):
        if self.home_pos_geo is None:
            self.home_pos_geo = msg.geo
            dropzone_vertices_local = [
                np.array(
                    geodetic2enu(
                        lat,
                        lng,
                        self.home_pos_geo.altitude,
                        self.home_pos_geo.latitude,
                        self.home_pos_geo.longitude,
                        self.home_pos_geo.altitude,
                    )
                )
                for lat, lng in self.gpx_tracks.airdrop_boundary
            ]
            self.get_logger().info(str(dropzone_vertices_local))
            for target_idx in range(4):
                target_pos = None
                while target_pos is None or any(
                    np.linalg.norm(target_pos - p) < 2 * TARGET_RADIUS
                    for p in self.target_positions_enu
                ):
                    target_pos = random_point_in_pca_rect(dropzone_vertices_local)
                self.target_positions_enu.append(target_pos)
                x, y, z = target_pos
                self._spawn_model(x, y, 0.2)

    def _drop_callback(self, request, response):
        if self.drops_done == 4:
            self.get_logger().info("Doing drops we don't have")
            return response
        drone_position = self.position_provider.get_position()
        self._spawn_model(
            drone_position[0],
            drone_position[1],
            drone_position[2] - 0.5,
            model_path=self.beacon_model_path,
        )
        for i in range(4):
            if (
                np.linalg.norm(drone_position[:2] - self.target_positions_enu[i][:2])
                < TARGET_RADIUS
            ):
                self.hit_counts[i] += 1
        self.drops_done += 1
        self._check_score()
        return response

    def _check_score(self):
        """
        Check if the mission has ended, then log final score
        """
        score = sum(min(x, 1) * 30 + 70 * x for x in self.hit_counts)
        self.get_logger().info(f"Hit Counts: {self.hit_counts}")
        self.get_logger().info(f"Score: {score}")
        with open(f"mission_score_{self.run_name}.txt", "w+") as f:
            f.write(f"{score}\n")

    def _spawn_model(self, x, y, z, model_path=None):
        # Random model
        if model_path is None:
            model_path = self.target_models[self.spawn_count % len(self.target_models)]
        model_name = f"target_{self.spawn_count}"
        self.spawn_count += 1

        # Spawn the model
        ld = spawn_model_launch_description(model_path, model_name, x, y, z)

        ls = LaunchService()
        ls.include_launch_description(ld)
        self.get_logger().info(f"Spawning {model_name} at ({x:.2f}, {y:.2f}, {z:.2f})")
        ls.run()


def main(args=None):
    suppress_stdout()  # Suppress annoying logs. This breaks 'print' but we can still use the logger
    rclpy.init(args=args)
    spawner = DynamicSpawner()
    rclpy.spin(spawner)
    spawner.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
