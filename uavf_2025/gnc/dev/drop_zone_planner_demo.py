#!/usr/bin/env python3

from gnc.commander_node import CommanderNode
import rclpy
import rclpy.node
import argparse
import time

# Commands to run:
# ros2 run uavf_2025 drop_zone_planner_demo.py src/uavf_2025/uavf_2025/gnc/data/main_field_west.gpx

if __name__ == "__main__":
    rclpy.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("gpx_file")
    args, unknown = parser.parse_known_args()

    node = CommanderNode(args)

    node.log("Starting to scan the dropzone")
    while True:
        node.log("Trying to get global position lock...")
        rclpy.spin_once(node)
        # Takeoff and wait to get to height
        node.takeoff()
        time.sleep(5)

        node.scan_dropzone()
        print("dropzone scan complete")

        node.land()
        break

    node.destroy_node()
    rclpy.shutdown()
