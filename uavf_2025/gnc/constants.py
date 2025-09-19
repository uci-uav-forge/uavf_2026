TAKEOFF_ALTITUDE = 16.0  # altitude in meters to takeoff to before starting waypoint lap
TARGET_ALTITUDE = 16.0  # altitude in meters we hover at when we drop over a target
SCAN_ALTITUDE = 16.0  # altitude in meters we perform the dropzone scan at

TAKEOFF_RADIUS = 1.0  # defines a radius in meters around our takeoff altitude where takeoff is considered complete
WP_RADIUS = 13.0  # defines a radius in meters around each waypoint where the waypoint is considered visited
TARGET_RADIUS = 1.0  # defines a radius in meters around each target waypoint where we are ready to drop
MAPPING_RADIUS = 3.0  # defines radius for mapping waypoints
TURN_LENGTH = (
    3.0  # in meters, the distance given to the copter to align itself with the dropzone
)

WP_LAP_SPEED = 11.2  # in m/s the speed through the waypoint lap and fly to targets
SCAN_SPEED = 2.5  # in m/s the speed through the drop zone scan
MAPPING_SPEED = 5.0

SCALE_TO_SLOW = 3  # distance in WP_RADIUSes when we begin to slow down for a turn

TARGET_GROUND_RADIUS = (
    1.0  # 7.0 distance we crop into the dropzone (update before comp.)
)

EMERGENCY_LAND_CH = 9  # channel designated for emergency landing when high
CH_THRESHOLD = 1900  # decides whether a channel is high or low
EMERGENCY_LAND_POINT = (38.315339, -76.548108)  # point determined by suas rules
