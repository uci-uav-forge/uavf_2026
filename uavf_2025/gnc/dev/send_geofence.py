import pymavlink.mavutil as utility
import pymavlink.dialects.v20.all as dialect
import argparse
from gnc.util import read_gpx_file, scale_geofence
from gnc.dev.start_mavros import find_pixhawk_port

# python3 dev/send_geofence.py data/your_file.gpx


def send_param(vehicle, name, value):
    message = dialect.MAVLink_param_set_message(
        target_system=vehicle.target_system,
        target_component=vehicle.target_component,
        param_id=name.encode(encoding="utf-8"),
        param_value=value,
        param_type=9,
    )  # float32 type

    vehicle.mav.send(message)
    message = vehicle.recv_match(
        type=dialect.MAVLink_param_value_message.msgname, blocking=True
    )


def main(geofence):
    ### GEOFENCE
    # geofence = [(33.642868, -117.826856),
    #             (33.642573, -117.826184),
    #             (33.64306, -117.825836),
    #             (33.643372, -117.826511)]

    # connect to vehicle
    port = find_pixhawk_port()
    if port:
        vehicle = utility.mavlink_connection(device=find_pixhawk_port(), baud=115200)
    else:
        vehicle = utility.mavlink_connection(device="udp:127.0.0.1:14550")

    # wait for a heartbeat
    vehicle.wait_heartbeat()

    # inform user
    print(
        "Connected to system:",
        vehicle.target_system,
        ", component:",
        vehicle.target_component,
    )

    vehicle.mav.heartbeat_send(
        utility.mavlink.MAV_TYPE_GCS, utility.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
    )  # Set to invalid for compenents that are not flight controllers

    ### Clear the fence mission (erase past geofences)
    print("Clearing the geofence...")
    message = dialect.MAVLink_mission_clear_all_message(
        target_system=vehicle.target_system,
        target_component=vehicle.target_component,
        mission_type=utility.mavlink.MAV_MISSION_TYPE_FENCE,
    )
    vehicle.mav.send(message)
    # Verify
    message = vehicle.recv_match(blocking=True)
    print(message)

    ### Send the new fence
    print("Sending mission_count...")
    message = dialect.MAVLink_mission_count_message(
        target_system=vehicle.target_system,
        target_component=vehicle.target_component,
        count=len(geofence),
        mission_type=utility.mavlink.MAV_MISSION_TYPE_FENCE,
    )
    vehicle.mav.send(message)

    print("Sending fence points...")
    seq = 0
    for vertex in geofence:
        message = vehicle.recv_match(blocking=True)
        print(message)
        message = dialect.MAVLink_mission_item_int_message(
            target_system=vehicle.target_system,
            target_component=vehicle.target_component,
            seq=seq,
            frame=utility.mavlink.MAV_FRAME_GLOBAL,
            command=utility.mavlink.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_INCLUSION,
            current=0,
            autocontinue=0,
            param1=len(geofence),
            param2=0.0,
            param3=0.0,
            param4=0.0,
            x=int(vertex[0] * 1e7),
            y=int(vertex[1] * 1e7),
            z=0,
            mission_type=utility.mavlink.MAV_MISSION_TYPE_FENCE,
        )
        vehicle.mav.send(message)
        seq += 1
    # Verify
    message = vehicle.recv_match(blocking=True)
    print(message)
    print("Done sending geofence. Now setting parameters...")

    send_param(vehicle, "FENCE_ENABLE", 1.0)
    send_param(vehicle, "FENCE_ACTION", 1.0)
    send_param(vehicle, "FENCE_TYPE", 5.0)
    send_param(vehicle, "FENCE_ALT_MAX", 150.0)
    send_param(vehicle, "OA_TYPE", 0.0)  # Uses djikstras

    print("Done setting geofence parameters.")
    print("Send geofence complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gpx_file")
    parser.add_argument(
        "--scale_factor", type=float, help="Factor to scale geofence by, default 1.0"
    )
    args, unknown = parser.parse_known_args()
    gpx_file = args.gpx_file
    print(gpx_file)
    if args.scale_factor is not None:
        scale_factor = args.scale_factor
    else:
        scale_factor = 1.0

    # Gets track
    tracks = read_gpx_file(gpx_file)
    geofence = tracks.flight_boundary

    # Scale geofence
    scaled_geofence = scale_geofence(geofence, scale_factor)
    scaled_geofence.pop()  # Get rid of duplicate point

    main(scaled_geofence)
