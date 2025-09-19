#!/usr/bin/env python3

from gnc.commander_node import CommanderNode
import rclpy
import rclpy.node
import argparse
from rclpy.executors import MultiThreadedExecutor
from threading import Thread

# ros2 run uavf_2025 diagnostic_check.py src/uavf_2025/uavf_2025/gnc/data/main_field_west.gpx

if __name__ == "__main__":
    print("Starting commander node...")

    rclpy.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("gpx_file")
    args, unknown = parser.parse_known_args()

    node = CommanderNode(args, preflight=True)

    node.get_logger().info("Starting CommanderNode executor...")
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    spinner = Thread(target=executor.spin)
    spinner.start()

    # Do diagnostic check
    node.log("Diagnostic check starting...")
    node.set_mode_and_verify("GUIDED")
    node.log("Diagnostic check finished.")

    node.pose_provider.disconnect()
    node.destroy_node()
    executor.shutdown()
