#!/usr/bin/env python3

# runs as a service with definition in /etc/systemd/system/coords_server.service

import time
import json
import gpxpy
import socket
import pathlib
import threading
import subprocess
from fastapi import FastAPI

import gpxpy.gpx
from collections import namedtuple
from typing import List

# BEGIN DUPLICATED CODE: SCUFFED BECAUSE ITS 3AM
TrackMap = namedtuple("TrackMap", ["mission"])


def read_gps(fname):
    with open(fname) as f:
        return [tuple(map(float, line.split(","))) for line in f]


class TrackNotFound(RuntimeError):
    def __init__(self, args=None) -> None:
        super().__init__("Can't find track name in GPX file")


def extract_coords_from_name(tracks: List[gpxpy.gpx.GPXTrack], label_suffix: str):
    """
    Returns the list of coordinates for a track with a name that ends with label_suffix.
    """

    trackNames = []
    for track in tracks:
        assert track.name, "No name given to track"
        trackNames.append(track.name)
        if track.name.endswith(label_suffix):
            coordinates = []
            for segment in track.segments:
                for point in segment.points:
                    if point.elevation:
                        coordinates.append(
                            (point.latitude, point.longitude, point.elevation)
                        )
                    else:
                        coordinates.append((point.latitude, point.longitude))
            return coordinates

    raise TrackNotFound(str(trackNames))


def read_gpx_file(file_name: str) -> TrackMap:
    """
    Return a named tuple for gpx tracks (A list of GPS coordinates). Attributes are mission, airdrop_boundary, flight_boundary, and
    mapping_boundary. The value is a list of GPS points that describe the associated track.
    """
    gpx_file = open(file_name, "r")
    gpx = gpxpy.parse(gpx_file)

    tracks = gpx.tracks
    track_map = TrackMap(
        extract_coords_from_name(tracks, "Mission"),
    )

    return track_map


# END DUPLICATED CODE


def connect_to_wifi(ssid, password):
    while True:
        try:
            # rescan for if network has just come up
            subprocess.run(["nmcli", "device", "wifi", "rescan"])
            command = ["nmcli", "device", "wifi", "connect", ssid, "password", password]

            result = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            print("Wifi connection successful:")
            print(result.stdout)
            return
        except subprocess.CalledProcessError as e:
            print("Failed to connect:")
            print(e.stderr)
            time.sleep(1)
            continue


def create_gpx(message):
    """
    Creates a new GPX file everytime coordinates are sent from
    the Ground Control Station. It currently replaces the existing
    copy of test.gpx.
    """
    target_dir = pathlib.Path(__file__).parent / "data"
    target_file = target_dir / "gcs_mission.gpx"
    location: str = message["data"]["location"]
    valid_locations = [
        "Runway 1",
        "Runway 2",
    ]
    if location not in valid_locations:
        print(f"Invalid location: {location}")
        return
    print(f"Creating GPX file for {location}...")
    src_name = "gcs_" + (location.lower()).replace(" ", "_") + ".gpx"
    src_file = target_dir / src_name
    print(f"Source file: {src_file}")
    with open(src_file, "r") as f:
        gpxpy_file = gpxpy.parse(f)
    mission = gpxpy.gpx.GPXTrack()
    mission.name = location + " Mission"
    segment = gpxpy.gpx.GPXTrackSegment()
    for waypoint in message["data"]["waypoints"]:
        lat = waypoint["lat"]
        lon = waypoint["lon"]
        ele = waypoint["alt"]
        gpx_waypoint = gpxpy.gpx.GPXTrackPoint(lat, lon, ele)
        segment.points.append(gpx_waypoint)
    mission.segments.append(segment)
    gpxpy_file.tracks.append(mission)
    gpxpy_file_str = gpxpy_file.to_xml()
    with open(target_file, "w") as f:
        f.write(gpxpy_file_str)
    print(f"Created {target_file} with {len(message['data']['waypoints'])} waypoints.")


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        try:
            connect_to_wifi("UAVForge", "coppcopp")
            result = subprocess.run(
                ["nmcli"], capture_output=True, text=True, check=True
            )

            str_to_search = result.stdout.strip()
            if "wifi" in str_to_search:
                str_to_search = str_to_search.split("wifi")[1]
            ip = str_to_search.split("inet4 ")[1].split("/")[0]

            s.close()
        except OSError:
            print("Failed to get own IP address, trying again in 1 second")
            time.sleep(1)
            continue
        return ip


def broadcast_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    ip = get_local_ip()
    print(f"[SERVER] Broadcasting IP: {ip}")
    while True:
        s.sendto(f"IP:{ip}".encode(), ("<broadcast>", 50000))
        time.sleep(1)


def start_broadcast_thread():
    """Start a thread to broadcast the local IP address."""
    broadcast_thread = threading.Thread(target=broadcast_ip)
    broadcast_thread.daemon = True
    broadcast_thread.start()
    print("Broadcast thread started.")


def connect_gcs():
    """Get the IP address of the GCS"""
    local_server_address = ("0.0.0.0", 37564)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(local_server_address)
    server_socket.listen(1)
    exit_flag = False
    while True:
        conn, addr = server_socket.accept()
        print(f"Connected to {addr[0]}")
        while True:
            try:
                data = conn.recv(4096).decode()
                if not data:
                    print("No data received, closing connection.")
                    conn.close()
                    break
                create_gpx(json.loads(data))
                print(f"Received data from {addr[0]}")
            except KeyboardInterrupt:
                print("Exiting...")
                exit_flag = True
                break
        if exit_flag:
            break
    print("Closing server socket...")
    server_socket.close()
    exit(0)


app = FastAPI()


@app.get("/waypoints")
def get_waypoints():
    print("got request for waypoints")
    track_map = read_gpx_file("data/gcs_mission.gpx")
    mission = track_map.mission
    print(mission)
    return mission


if __name__ == "__main__":
    app.add_api_route("get_waypoints", get_waypoints)
    print("waiting")
