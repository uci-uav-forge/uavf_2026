import pymavlink.mavutil as utility
from gnc.dev.start_mavros import find_pixhawk_port
import pymavlink.dialects.v20.all as dialect
import time


def main():
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

    waiting = True
    while waiting:
        vehicle.mav.heartbeat_send(
            utility.mavlink.MAV_TYPE_QUADROTOR,
            utility.mavlink.MAV_AUTOPILOT_INVALID,
            0,
            0,
            0,
        )  # Set to invalid for compenents that are not flight controllers

        # Verify
        message = vehicle.recv_match(type="HEARTBEAT", blocking=True)
        if message.type == 2:
            print(message)
        if message.type == 2 and message.system_status == 5:
            print("possible rc failsafe...")
            waiting = False

    time.sleep(150)

    msg = dialect.MAVLink_rc_channels_override_message(
        target_system=vehicle.target_system,
        target_component=vehicle.target_component,
        chan1_raw=0,
        chan2_raw=0,
        chan3_raw=0,
        chan4_raw=0,
        chan5_raw=0,
        chan6_raw=2000,
        chan7_raw=0,
        chan8_raw=0,
    )
    vehicle.mav.send(msg)


if __name__ == "__main__":
    main()
