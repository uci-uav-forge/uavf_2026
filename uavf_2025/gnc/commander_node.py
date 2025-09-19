# imports for logging
import logging
import os

# imports other
import time
import warnings
from pathlib import Path
from time import strftime

# imports mavros
import mavros_msgs.msg
import mavros_msgs.srv

# import rclpy
import numpy as np
import pymap3d as pm
import rclpy
import rclpy.node
import rclpy.time
from rcl_interfaces.srv import SetParameters
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
import mavros_msgs.srv._param_set
from mavros_msgs.msg import FullSetpoint
import mavros_msgs.srv

# imports other
from gnc.util import validate_points, read_gpx_file, calculate_alignment_yaw
from gnc.geofence import Geofence

from shared import ensure_ros

from gnc.dropzone_planner import DropzonePlanner
from gnc.util import (
    gps_to_local,
)

from shared import PDBMonitor, DronePoseProvider, UtilizationLogger
from shared.types import GlobalPosition


# imports perception
from perception import Perception
from perception.lib.util import create_console_logger

from .constants import (
    MAPPING_RADIUS,
    SCALE_TO_SLOW,
    TAKEOFF_ALTITUDE,
    TAKEOFF_RADIUS,
    TARGET_ALTITUDE,
    TARGET_RADIUS,
    WP_LAP_SPEED,
    WP_RADIUS,
    TARGET_GROUND_RADIUS,
    EMERGENCY_LAND_CH,
    CH_THRESHOLD,
    EMERGENCY_LAND_POINT,
)
from .routing import Routing


class CommanderNode(rclpy.node.Node):
    """
    Manages subscriptions to ROS2 topics and services necessary for the main GNC node.
    """

    @ensure_ros  # Decorator to start rclpy if it hasn't already
    def __init__(self, args=None, preflight=False, run_avoidance=False):
        super().__init__("uavf_commander_node")

        # logging configs
        self.src_path = str(Path(__file__).absolute().parent.parent.parent.parent)
        logs_base = Path(f"{self.src_path}/uavf_2025/logs")
        year_month_day = strftime("%Y-%m-%d")
        index = len(list((logs_base / year_month_day).glob("*-*-*-*")))
        time_string = strftime("%Y-%m-%d/%H-%M")
        self.logs_path = logs_base / f"{index}-{time_string}"
        self.logs_path.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=str(self.logs_path / "commander_node.log"),
            format="%(asctime)s %(message)s",
            encoding="utf-8",
            level=logging.DEBUG,
        )
        logging.getLogger().addHandler(logging.StreamHandler())

        # Quality of Service config
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_ALL,
            depth=1,
        )

        # Create all the mavros clients, publishers and subscribers
        self._mavros_topic_init(qos_profile)
        self.channels = []

        self.wait_for_initial_state()  # Wait for initial arm and mode state
        self.last_wp_seq = 0  # Subscription to get when waypoint is reached

        # initialize drone pose provider and wait for it to be ready
        # MUST BE BEFORE DROPZONE PLANNER AND HOME POSE INIT
        self.log("Creating drone pose provider...")
        self._logger = create_console_logger(self.logs_path, "gnc")

        self._utilization_logger = UtilizationLogger(
            Path(self.logs_path / "system_utilization.csv"),
            period=0.5,
        )
        self._utilization_logger.start()

        self.pose_provider = DronePoseProvider(
            self._logger, self.logs_path / "drone_pose"
        )
        self.pose_provider.wait_until_ready(self)

        # Initialize the home position
        self.global_home = GlobalPosition()
        self.get_home_position()

        # import gpx
        # Load the coordinate points for waypoints, boundaries
        self.log("Loading gpx file...")
        if args:
            self.gpx_track_map = read_gpx_file(args.gpx_file)
            self.auto_arm = args.auto_arm
        else:
            raise RuntimeError("Required argument <gpx_file>")

        self.mission_wps, self.dropzone_bounds, self.geofence, self.mapping_bounds = (
            self.gpx_track_map.mission,
            self.gpx_track_map.airdrop_boundary,
            self.gpx_track_map.flight_boundary,
            self.gpx_track_map.mapping_boundary,
        )
        self.mission_wps = Routing.generate_global_trajectory(
            self.mission_wps, self.geofence, self.global_home
        )
        self.log(f"Dropzone Bounds: {self.dropzone_bounds}")
        if not preflight:
            validate_points(self.mission_wps, self.geofence)
            validate_points(self.dropzone_bounds, self.geofence, False)
            validate_points(self.mapping_bounds, self.geofence, False)
            self.log("\n ========== GPX FILE VALIDATED ============== \n")

        # Initialize Commander ARM to false
        self.armed = False

        # Initialize dropzone planner
        self.dropzone_planner = DropzonePlanner(self)

        # instantiating Perception instance
        self.perception = Perception(
            Perception.CameraType.AUTO,
            self.logs_path,
            pose_provider_override=self.pose_provider,
            enable_tracking=run_avoidance,
            enable_mapping=True,
            mapping_path=Path(f"{logs_base}/usb/{time_string}"),
            mapping_roi=np.array(
                [[*coord, 0] for coord in self.get_roi_corners_local()]
            ),
        )

        self.log(f"===== Instantiate Image Mapping {self.perception._mapper}")

        self.pulled_params = False

        self.pdb_monitor = PDBMonitor(self)

        self.speed: float | None = WP_LAP_SPEED

        self.emergency_landing = False

        self.log("Finished init")

    def _mavros_topic_init(self, qos_profile) -> None:
        """
        All mavros subscriptions, publishers and clients to be instantiated on construction.
        """
        # Mavros services
        self.clear_mission_client = self.create_client(
            mavros_msgs.srv.WaypointClear, "mavros/mission/clear"
        )
        self.waypoints_client = self.create_client(
            mavros_msgs.srv.WaypointPush, "mavros/mission/push"
        )
        self.mode_client = self.create_client(
            mavros_msgs.srv.SetMode, "mavros/set_mode"
        )
        self.arm_client = self.create_client(
            mavros_msgs.srv.CommandBool, "mavros/cmd/arming"
        )
        self.takeoff_client = self.create_client(
            mavros_msgs.srv.CommandTOL, "mavros/cmd/takeoff"
        )
        self.command_int_client = self.create_client(
            mavros_msgs.srv.CommandInt, "mavros/cmd/command_int"
        )
        self.param_push_client = self.create_client(
            mavros_msgs.srv.ParamPush, "mavros/param/push"
        )
        self.param_pull_client = self.create_client(
            mavros_msgs.srv.ParamPull, "mavros/param/pull"
        )
        self.param_setter_client = self.create_client(
            SetParameters, "/mavros/param/set_parameters"
        )
        self.req = SetParameters.Request()
        self.state_subscriber = self.create_subscription(
            mavros_msgs.msg.State,
            "/mavros/state",
            self.state_callback,
            qos_profile,
        )
        self.msg_pub = self.create_publisher(
            mavros_msgs.msg.StatusText, "mavros/statustext/send", qos_profile
        )
        self.pos_vel_pub = self.create_publisher(
            FullSetpoint, "/mavros/setpoint_full/cmd_full_setpoint", qos_profile
        )
        self.ch_sub = self.create_subscription(
            mavros_msgs.msg.RCIn, "/mavros/rc/in", self.ch_callback, qos_profile
        )

    def log(self, *args, **kwargs) -> None:
        """
        Log the given arguments.
        """
        logging.info(*args, **kwargs)

    def log_status(self, msg, announce=True):
        self.msg_pub.publish(
            mavros_msgs.msg.StatusText(
                severity=(
                    mavros_msgs.msg.StatusText.NOTICE
                    if announce
                    else mavros_msgs.msg.StatusText.INFO
                ),
                text=msg,
            )
        )

    def change_speed(self, meters_per_second: float | None = None):
        self.speed = meters_per_second

    def get_home_position(self) -> None:
        """
        Sets self.local_home to the home position of the drone.
        Gets home position based off ekf origin and not ardupilot home.
        This ensures that local position setpoints are geographically accurate.
        """
        self.log("========  GETTING HOME POSITION ============")
        # get local pose from Pose Provider
        current_local_pose = self.pose_provider.get_local_pose()
        current_global_pose = self.pose_provider.get_global_pose()

        # geodetic2enu(current_global_pose, self.global_home) = current_local_pose
        # solve for self.global_home
        global_home_tuple = pm.enu2geodetic(
            -current_local_pose.position[0],
            -current_local_pose.position[1],
            -current_local_pose.position[2],
            current_global_pose.latitude,
            current_global_pose.longitude,
            current_global_pose.altitude,
        )

        inaccuracy = np.linalg.norm(
            np.array(
                pm.geodetic2enu(
                    current_global_pose.latitude,
                    current_global_pose.longitude,
                    current_global_pose.altitude,
                    *global_home_tuple,
                )
            )
            - np.array(current_local_pose.position)
        )
        if inaccuracy > 0.01:
            warnings.warn(
                f"Home position calculation is off by more than 1cm: {inaccuracy} meters"
            )

        self.global_home = GlobalPosition(*global_home_tuple)

        self.log(f"========== HOME INITIALIZED: {self.global_home} ========")

    def get_dropzone_bounds_mlocal(self) -> list[tuple[float, float]]:
        """
        Returns the bounds of the drop zone in local coordinates.
        """
        return [gps_to_local(x, self.global_home) for x in self.dropzone_bounds]

    def get_roi_corners_local(self):
        """
        Return the top left, top right, and bottom left corners of the ROI in local coordinates
        (in that order).
        """
        return (
            gps_to_local(self.mapping_bounds[2], self.global_home),
            gps_to_local(self.mapping_bounds[3], self.global_home),
            gps_to_local(self.mapping_bounds[1], self.global_home),
        )

    def get_pose_position(self) -> np.ndarray:
        """
        Return the drone's current local position.
        """
        return self.pose_provider.get_local_pose().position

    def scan_dropzone(self) -> None:
        """
        Scans the dropzone uding the scan_dropzone function in
        dropzone_planner.
        """
        self.dropzone_bounds_mlocal = self.get_dropzone_bounds_mlocal()
        self.log(f"dropzone bounds = {self.dropzone_bounds_mlocal}")

        self.change_speed(None)

        self.log("Scanning Dropzone")
        self.dropzone_planner.scan_dropzone()

    def execute_waypoints(
        self,
        waypoints: list[tuple[float, float, float]],
        mapping=False,
        after_second_waypoint_cb=None,
        set_yaw=False,
    ) -> None:
        """
        Flys a single waypoint lap along provided list of waypoints

        parameters:
            waypoints - expects a list of tuple(float, float, float) each being (latitude, longitude, altitude)
            mapping - boolean, default false, if true will not convert coords to local and will fly to points directly, smaller wp radius
            after_second_waypoint - Optional, a function that will execute once after the second waypoint is reached
            set_yaw - yaws drone to face waypoints
        returns:
            None
        """
        self.log("Executing waypoint lap..")

        # Will raise an error if waypoints are outside of geofence
        if mapping:
            validate_points(
                waypoints,
                [
                    gps_to_local(fence_point, self.global_home)
                    for fence_point in self.geofence
                ],
            )
        else:
            validate_points(waypoints, self.geofence)

        self.log(f"Waypoint plan approved: {waypoints}")
        nominal_speed = self.speed
        for index, gps_wp in enumerate(waypoints):
            self.log(f"Flying to waypoint {index + 1}/{len(waypoints)}: {gps_wp}")
            self.log_status(f"Flying to waypoint {index + 1}/{len(waypoints)}")

            if not mapping:
                # convert wp into local frame
                wp = np.array(
                    [*gps_to_local((gps_wp[0], gps_wp[1]), self.global_home), gps_wp[2]]
                )
            else:
                wp = np.array(gps_wp)

            curr_position = self.get_pose_position()
            log_timer = time.time()
            setpoint_timer = time.time()
            self.fly_to_position(wp, set_yaw=set_yaw)
            while np.linalg.norm(curr_position - wp) > (
                WP_RADIUS if not mapping else MAPPING_RADIUS
            ):
                curr_position = self.get_pose_position()
                self.pdb_monitor.rtl_time(
                    (curr_position[0] ** 2 + curr_position[1] ** 2) ** 0.5,
                    curr_position[2],
                )

                # Slowly decrease speed on approach of wp radius for waypoint lap
                dist = np.linalg.norm(curr_position - wp)
                if nominal_speed == WP_LAP_SPEED and dist < WP_RADIUS * (
                    SCALE_TO_SLOW + 1
                ):
                    self.change_speed(
                        max(
                            (float(dist) - WP_RADIUS)
                            / (SCALE_TO_SLOW * WP_RADIUS)
                            * (WP_LAP_SPEED / 2)
                            + (WP_LAP_SPEED / 2),
                            WP_LAP_SPEED / 2,
                        )
                    )
                else:
                    self.change_speed(nominal_speed)

                if mapping:
                    horiz = (curr_position[0] ** 2 + curr_position[1] ** 2) ** 0.5
                    vert = curr_position[2]
                    failafe = self.pdb_monitor.battery_failsafe(horiz, vert)
                    if failafe:
                        self.return_to_launch()
                        self.log("Aborting mapping due to battery failsafe...")
                        self.log_status("Aborting mapping due to battery failsafe...")
                        self.perception._mapper.save_map()
                        while (
                            True
                        ):  # Enter an inifinite loop to keep logging while aborting
                            pass

                # Log distance from waypoint every 3 seconds
                if time.time() - log_timer > 3:
                    self.log(
                        f"Distance from waypoint is {np.linalg.norm(curr_position - wp)} meters..."
                    )
                    log_timer = time.time()

                if time.time() - setpoint_timer > 0.1:
                    self.fly_to_position(wp, set_yaw=set_yaw)
                    setpoint_timer = time.time()

            self.change_speed(nominal_speed)
            if (
                index + 1 == 2 and after_second_waypoint_cb is not None
            ):  # Check if the second waypoint has been reached
                self.log("Executing custom second waypoint callback function...")
                after_second_waypoint_cb()
                nominal_speed = self.speed

            # have last waypoint not be velocity based
            if index == len(waypoints) - 1:
                self.change_speed(0.5)
                self.fly_to_position(wp, set_yaw=set_yaw)

        self.log("Execute waypoints complete")

    def takeoff(self) -> None:
        """
        Set to GUIDED mode, arm the drone, and then takeoff using self.takeoff_client.
        """
        self.log("Attempting takeoff...")

        self.set_mode_and_verify("STABILIZE")  # Can't arm in guided

        self.arm()

        # must reset the mode to guided for possible controller override on manual arming
        self.set_mode_and_verify("GUIDED")

        self.log("Checking we aren't taking off outside the geofence.")
        global_pose = self.pose_provider.get_global_pose()
        validate_points(
            [(global_pose.latitude, global_pose.longitude)],
            self.geofence,
            has_altitudes=False,
        )

        self.takeoff_client.call(
            mavros_msgs.srv.CommandTOL.Request(
                min_pitch=float("NaN"),
                yaw=float("NaN"),
                latitude=float("NaN"),
                longitude=float("NaN"),
                altitude=float(TAKEOFF_ALTITUDE),
            )
        )

        self.log(f"Requested takeoff: {TAKEOFF_ALTITUDE}")

        # self.waypoints_reached_local.append(self.get_pose_position())
        while abs(self.get_pose_position()[2] - TAKEOFF_ALTITUDE) > TAKEOFF_RADIUS:
            time.sleep(1)
            self.log(f"Taking Off... Current altitude: {self.get_pose_position()[2]}")

        self.perception.start_recording()

    def arm(self) -> None:
        """
        Runs until the drone is armed.
        """
        self.log_status("Jetson says hi...")
        self.set_ardupilot_home_pos()  # So ardupilot can RTL to its last arm position
        self.log("Attempting arming...")
        self.pdb_monitor.start_logging()

        while True:
            self.log(f"Self.armed = {self.armed}, self.auto_arm = {self.auto_arm}")
            # only arm if drone state armed or user specified autoarm.
            if self.auto_arm:
                self.arm_client.call(mavros_msgs.srv.CommandBool.Request(value=True))
            if self.armed:
                break
            else:
                self.log("Waiting for drone to be armed...")
            time.sleep(1)

        self.log("Drone armed.")

    def land(self) -> None:
        """
        Execute land command.
        """
        self.log("Attempting landing...")
        self.set_mode_and_verify("LAND")

    def return_to_launch(self) -> None:
        """
        Execute return to launch command.
        """
        self.log("Attempting return to launch...")
        self.set_mode_and_verify("RTL")

        # no need to do this if we want to log RTL efficiency
        # self.pdb_monitor.stop_logging()

    def fly_to_target_local(
        self, point_x: float, point_y: float, point_z: float, index: int
    ) -> None:
        """
        Fly to a target that is specified in local coordinates.
        """
        # set perception lock to current target (no gimbal cam for now)
        # self.perception.set_target_lock(index)

        self.log(f"Fly to local target ({point_x}, {point_y}, {point_z})")

        # send updated set positon until we reach the waypoint
        target_position = np.array([point_x, point_y, point_z])
        setpoint_timer = time.time()

        while (
            np.linalg.norm(self.get_pose_position() - target_position) > TARGET_RADIUS
        ):
            if time.time() - setpoint_timer > 0.1:
                if index != -1:
                    self.perception.set_target_lock(index)
                    updated_tracks = self.filter_bad_targets(
                        self.perception.get_target_positions()
                    )
                    self.log(
                        f"Current Position: {self.get_pose_position()}, target: {target_position}"
                    )

                    point = updated_tracks[index].position.copy()
                    self.log(f"Updated Target Position: {point}")

                    point[2] = TARGET_ALTITUDE  # Set z axis to above the target
                else:
                    self.log("index -1 target")
                    point = [point_x, point_y, point_z]

                # set reposition=True to rotate to look toward target
                self.fly_to_position(point, set_yaw=True)
                target_position = np.array(point)
                setpoint_timer = time.time()

    def fly_to_targets_in_dropzone(self) -> None:
        """
        Fly to targets in the drop zone.
        """
        self.change_speed(None)
        self.dropzone_planner.target_path()
        self.perception.set_idle()
        self.change_speed(WP_LAP_SPEED)

    def set_mode_and_verify(self, mode: str) -> None:
        """
        Sets the mode of the drone and waits until we can verify it has been changed.
        Make sure param mode is an official mode.
        """
        self.log(f"Attempting to set mode to {mode}...")
        self.mode_client.call(
            mavros_msgs.srv.SetMode.Request(base_mode=0, custom_mode=mode),
        )
        self.log(f"Verifying mode: {mode}...")
        while self.mode.lower() != mode.lower():
            self.mode_client.call_async(
                mavros_msgs.srv.SetMode.Request(base_mode=0, custom_mode=mode),
            )
        self.log("Mode change verified.")

    def wait_for_initial_state(self) -> None:
        """
        Waits for the initial state of arm and mode and stops waiting
        when mode is not None.
        """
        self.log("Waiting for initial state of arm and mode...")
        self.mode = "None"
        while self.mode == "None" or len(self.channels) == 0:
            rclpy.spin_once(self)
            time.sleep(0.5)
        self.log("Initial state received.")

    def state_callback(self, msg: mavros_msgs.msg.State) -> None:
        """
        Callback function to get the state of the drone from
        Mavros.
        """
        self.armed: bool = msg.armed
        self.mode: str = msg.mode

    def set_ardupilot_home_pos(self):
        """
        Sends a message to ardupilot to consider its curruent positon its home positon.
        This only affects RTL's target landing point.
        """
        self.log(
            "Telling ardupilot to consider its current position is its home/RTL point..."
        )
        self.command_int_client.call(
            mavros_msgs.srv.CommandInt.Request(
                frame=0,  # Global
                command=179,  # MAV_CMD_DO_SET_HOME
                current=0,
                autocontinue=0,
                param1=1.0,  # Consider current position
                param2=0.0,
                param3=0.0,
                param4=0.0,
                x=0,
                y=0,
                z=0.0,
            )
        )
        self.log("Ardupilot home position set to current position.")

    def fly_to_position(
        self,
        target_position: np.ndarray,
        set_yaw: bool = False,
    ):
        """
        Flies to a local position given by (x, y, z) using the NED local frame.
        If reposition is true, the drone will yaw itself to point at the target position.
        Speed is optional and defaults to the waypoint lap speed; units are m/s.
        """

        if (
            self.mode.upper() != "GUIDED"
        ):  # Allows manual takeover in the middle of a setpoint
            self.log(f"Error: sending position while in {self.mode}...")
            self.log_status(f"Error: sending position while in {self.mode}...")
            return

        if self.emergency_landing or self.check_emergency_land():
            self.change_speed(None)
            set_yaw = False
            if not self.emergency_landing:
                self.log("Emergency land trigerred...")
                self.log_status("Emergency land trigerred...")
                self.emergency_landing = True
                self.execute_waypoints([(*EMERGENCY_LAND_POINT, TARGET_ALTITUDE)])
                self.set_mode_and_verify("LAND")
                while True:  # get into an infinite loop to keep logging while landing
                    pass

        pxyz = (
            (target_position[0], target_position[1], target_position[2])
            if self.speed is None
            else None
        )
        curr_position = self.get_pose_position()
        vxyz = None
        yaw = None

        if self.speed is not None:
            # Calculate velocity
            norm = (target_position - curr_position) / np.linalg.norm(
                target_position - curr_position
            )
            norm *= self.speed
            vxyz = (norm[0], norm[1], norm[2])

        if set_yaw:
            yaw = calculate_alignment_yaw(curr_position, target_position)

        self.set_full_setpoint(pxyz=pxyz, vxyz=vxyz, yaw=yaw)

    def map(self):
        """
        Map all currently unmapped regions in ROI.
        """
        self.perception.set_idle()
        self.dropzone_planner.mapping_path(45, 15.0, 1.0)

    def map_tsp(self):
        self.perception.set_idle()
        self.dropzone_planner.mapping_path_tsp(45.72, 15.0, 1.0)

    def _build_setpoint_typemask(
        self,
        pxyz: tuple[float, float, float] | np.ndarray | None,
        vxyz: tuple[float, float, float] | np.ndarray | None,
        axyz: tuple[float, float, float] | np.ndarray | None,
        yaw: float | None,
        yaw_rate: float | None,
    ):
        # https://mavlink.io/en/messages/common.html#POSITION_TARGET_TYPEMASK
        typemask = 0
        if pxyz is not None:
            if isinstance(pxyz, tuple) or isinstance(pxyz, list):
                pxyz = np.array([np.nan if i is None else i for i in pxyz])
        else:
            pxyz = np.array([np.nan, np.nan, np.nan])
        for i in range(3):
            if np.isnan(pxyz[i]):
                typemask |= 1 << i

        if vxyz is not None:
            if isinstance(vxyz, tuple) or isinstance(vxyz, list):
                vxyz = np.array([np.nan if i is None else i for i in vxyz])
        else:
            vxyz = np.array([np.nan, np.nan, np.nan])
        for i in range(3):
            if np.isnan(vxyz[i]):
                typemask |= 1 << (i + 3)

        if axyz is not None:
            if isinstance(axyz, tuple) or isinstance(axyz, list):
                axyz = np.array([np.nan if i is None else i for i in axyz])
        else:
            axyz = np.array([np.nan, np.nan, np.nan])
        for i in range(3):
            if np.isnan(axyz[i]):
                typemask |= 1 << (i + 6)

        yaw = float(yaw) if yaw is not None else np.nan
        if np.isnan(yaw):
            typemask |= 1 << 10

        yaw_rate = np.nan if yaw_rate is None else float(yaw_rate)
        if np.isnan(yaw_rate):
            yaw_rate = 0.0  # if yawrate is not passed, set it as 0 so its "ignored"

        return typemask, pxyz, vxyz, axyz, yaw, yaw_rate

    def set_full_setpoint(
        self,
        pxyz: tuple[float, float, float] | np.ndarray | None = None,
        vxyz: tuple[float, float, float] | np.ndarray | None = None,
        axyz: tuple[float, float, float] | np.ndarray | None = None,
        yaw: float | None = None,
        yaw_rate: float | None = None,
        *,
        typemask: int | None = None,
    ):
        """
        Sets the full setpoint of the drone.
        WORKS
        """
        new_typemask, pxyz, vxyz, axyz, yaw, yaw_rate = self._build_setpoint_typemask(
            pxyz, vxyz, axyz, yaw, yaw_rate
        )
        if typemask is None:
            typemask = new_typemask
        if typemask == 3583:  # 0b110111111111
            return  # all values are None
        data = FullSetpoint()
        data.header.stamp = self.get_clock().now().to_msg()
        data.type_mask = typemask
        data.position.x = float(pxyz[0])
        data.position.y = float(pxyz[1])
        data.position.z = float(pxyz[2])
        data.velocity.x = float(vxyz[0])
        data.velocity.y = float(vxyz[1])
        data.velocity.z = float(vxyz[2])
        data.acceleration.x = float(axyz[0])
        data.acceleration.y = float(axyz[1])
        data.acceleration.z = float(axyz[2])
        data.yaw = float(yaw)
        data.yaw_rate = float(yaw_rate)
        self.pos_vel_pub.publish(data)

    def filter_bad_targets(self, tracks: list):
        target_radius = TARGET_GROUND_RADIUS
        bounds = self.get_dropzone_bounds_mlocal()

        fence = Geofence(bounds).shrink(target_radius)

        new_tracks = [
            track
            for track in tracks
            if fence.contains((track.position[0], track.position[1]))
        ]

        return new_tracks

    def check_emergency_land(self):
        """
        Checks if the emergency land channel is high.
        """
        if len(self.channels) == 0:
            return False
        ch = self.channels[EMERGENCY_LAND_CH - 1]
        # self.log(f"channel: {ch}")
        return ch > CH_THRESHOLD and os.environ.get("ENABLE_RC", False)

    def ch_callback(self, msg: mavros_msgs.msg.RCIn):
        """Callback function for rc-in channels"""
        self.channels = msg.channels
        # self.log(f"channel callback: {msg.channels}")
