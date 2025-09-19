import pymavlink.mavutil as utility
import pymavlink.dialects.v20.all as dialect

# Run as any python file: python3 clean_mission.py
# Change line 8 device to switch connection

# connect to vehicle
connection = utility.mavlink_connection(device="udp:127.0.0.1:14550")


# wait for a heartbeat
connection.wait_heartbeat()


# inform user
print(
    "Connected to system:",
    connection.target_system,
    ", component:",
    connection.target_component,
)

print("Clearing the geofence...")
message = dialect.MAVLink_mission_clear_all_message(
    target_system=connection.target_system,
    target_component=connection.target_component,
    mission_type=dialect.MAV_MISSION_TYPE_FENCE,
)
connection.mav.send(message)
# Verify
message = connection.recv_match(
    type=dialect.MAVLink_mission_ack_message.msgname, blocking=True
)
print(f"Cleared the geofence: {message}")

print("Clearing main mission items...")  # Waypoints, takeoffs, landings, etc.
message = dialect.MAVLink_mission_clear_all_message(
    target_system=connection.target_system,
    target_component=connection.target_component,
    mission_type=dialect.MAV_MISSION_TYPE_MISSION,
)
connection.mav.send(message)
# Verify
message = connection.recv_match(
    type=dialect.MAVLink_mission_ack_message.msgname, blocking=True
)
print(f"Cleared main mission items: {message}")

print("Clearing rally points...")
message = dialect.MAVLink_mission_clear_all_message(
    target_system=connection.target_system,
    target_component=connection.target_component,
    mission_type=dialect.MAV_MISSION_TYPE_RALLY,
)
connection.mav.send(message)
# Verify
message = connection.recv_match(
    type=dialect.MAVLink_mission_ack_message.msgname, blocking=True
)
print(f"Cleared the rally points: {message}")


# Reset fence params
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


print("Disabling fence and fence avoidance...")
send_param(connection, "FENCE_ENABLE", 0.0)  # Disables fence
send_param(connection, "OA_TYPE", 0.0)  # Disables djikstras
print("Clean complete.")

message = dialect.MAVLink_statustext_message(severity=0, text="blud".encode("utf-8"))
connection.mav.send(message)
