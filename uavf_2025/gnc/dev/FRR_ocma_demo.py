#!/usr/bin/env python3

from gnc.commander_node import CommanderNode
import rclpy
import rclpy.node
import argparse
from rclpy.executors import MultiThreadedExecutor
from threading import Thread
import os
from time import sleep

if __name__ == "__main__":
    print("Starting commander node...")

    rclpy.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("gpx_file")
    parser.add_argument(
        "--auto-arm", action="store_true", help="Automatically arm without user prompt"
    )
    args, unknown = parser.parse_known_args()

    do_run_avoidance = bool(os.getenv("RUN_AVOIDANCE", False))

    node = CommanderNode(args, run_avoidance=do_run_avoidance)

    node.get_logger().info("Starting CommanderNode executor")
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    spinner = Thread(target=executor.spin)
    spinner.start()

    while True:
        try:
            if node.pose_provider.get_local_pose() is not None:
                break
        except BufferError:
            pass
        sleep(1)

    node.log("Starting waypoint loop")
    while True:
        # Takeoff and wait to get to height
        if not node.armed:
            node.takeoff()

        if not os.getenv("SKIP_WAYPOINTS", False):
            for i in range(10):
                node.execute_waypoints(node.mission_wps)
            node.log("waypoints complete")

        node.return_to_launch()
        break
    node.perception.cleanup()
    node.pose_provider.disconnect()
    quit()
