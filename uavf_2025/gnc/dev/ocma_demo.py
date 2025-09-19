#!/usr/bin/env python3

from gnc.commander_node import CommanderNode
import rclpy
import rclpy.node
import argparse
import time
from rclpy.executors import MultiThreadedExecutor
from threading import Thread
# Commands to run:
# ros2 run uavf_2025 ocma_demo.py src/uavf_2025/uavf_2025/gnc/data/main_field_west.gpx
# OR
# ros2 launch uavf_2025 commander_node.launch.py gpx_file:=src/uavf_2025/uavf_2025/gnc/data/main_field_west.gpx

if __name__ == "__main__":
    print("Starting commander node...")

    rclpy.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("gpx_file")
    args, unknown = parser.parse_known_args()

    node = CommanderNode(args)
    node.get_logger().info("Starting waypoint loop")

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    spinner = Thread(target=executor.spin)
    spinner.start()

    node.log("Starting waypoint loop")
    while True:
        # Takeoff and wait to get to height
        node.arm()
        time.sleep(10)
        break

    node.pose_provider.disconnect()
    node.destroy_node()
    executor.shutdown()
