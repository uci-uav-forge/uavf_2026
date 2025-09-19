import gpxpy
import gpxpy.gpx
import numpy as np
from numpy import typing as npt
from shapely.geometry import Point, Polygon
from collections import namedtuple
from typing import List
import pymap3d as pm
from shared.types import GlobalPosition
from shapely.affinity import scale

# imports for typing
import geometry_msgs.msg


TrackMap = namedtuple(
    "TrackMap", ["mission", "airdrop_boundary", "flight_boundary", "mapping_boundary"]
)


def scale_geofence(points: list[tuple[float, float]], scale_factor: float):
    """
    Scales a geofence polygon in global coordinates (lat, lon) by converting to local ENU,
    applying scaling, and converting back.

    :param points: List of (lat, lon) tuples representing the geofence vertices.
    :param scale_factor: Scaling factor (1.0 means no change, >1 expands, <1 contracts).
    :return: List of (lat, lon) tuples representing the modified geofence vertices.
    """
    if len(points) < 3:
        raise ValueError("A polygon must have at least 3 points.")

    # Choose an arbitrary origin (first point in list)
    lat0, lon0 = points[0]

    # Convert lat/lon to local ENU coordinates
    arbitrary_origin = GlobalPosition(
        latitude=points[0][0], longitude=points[0][1], altitude=0.0
    )
    local_coords = [gps_to_local(pt, arbitrary_origin) for pt in points]

    # Create polygon in local coordinates
    polygon = Polygon(local_coords)

    # Scale the polygon around its centroid
    scaled_polygon = scale(
        polygon, xfact=scale_factor, yfact=scale_factor, origin="centroid"
    )

    # Convert back to global coordinates
    finished_local_coords = scaled_polygon.exterior.coords[:]
    scaled_points = [local_to_gps(pt, arbitrary_origin) for pt in finished_local_coords]  # type: ignore

    return scaled_points


def read_gps(fname):
    with open(fname) as f:
        return [tuple(map(float, line.split(","))) for line in f]


def is_point_within_fence(point, geofence):
    fence_polygon = Polygon(geofence)
    point = Point(point)
    return fence_polygon.contains(point)


def validate_points(point_list, geofence, has_altitudes=True):
    for point in point_list:
        if has_altitudes:
            assert (
                len(point) == 3
            ), "ERROR: Point does not contain all three: Lat, Lon, Alt."
            assert point[2] > 0, "ERROR: Altitude must be greater than 0."
        else:
            assert len(point) == 2
        assert is_point_within_fence(
            (point[0], point[1]), geofence
        ), f"ERROR: Point ({point[0]}, {point[1]}) is not within Geofence: {geofence}."
    return


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
        extract_coords_from_name(tracks, "Airdrop Boundary"),
        extract_coords_from_name(tracks, "Flight Boundary"),
        extract_coords_from_name(tracks, "Mapping Boundary"),
    )

    return track_map


def is_inside_bounds_local(bounds, pt):
    p = Point(pt[0], pt[1])
    boundary = Polygon(bounds)

    return p.within(boundary)


def calculate_turn_angles_deg(path_coordinates):
    norm_vectors = []

    for i in range(len(path_coordinates) - 1):
        tail_x, tail_y = path_coordinates[i][0], path_coordinates[i][1]
        head_x, head_y = path_coordinates[i + 1][0], path_coordinates[i + 1][1]

        result_vector = np.array([head_x - tail_x, head_y - tail_y])
        norm_vectors.append(result_vector / np.linalg.norm(result_vector))

    turn_angles = []
    for i in range(len(norm_vectors) - 1):
        turn_angles.append(
            np.degrees(np.arccos(np.dot(norm_vectors[i], norm_vectors[i + 1])))
        )

    return turn_angles


def pose_to_xy(pose: geometry_msgs.msg.PoseStamped) -> npt.NDArray[np.float64]:
    pos = pose.pose.position
    return np.array([pos.x, pos.y])


def local_to_gps(
    local: tuple[float, float], global_home: GlobalPosition
) -> tuple[float, float]:
    """
    `local` is 2d XY
    """
    return pm.enu2geodetic(
        *local,
        0,
        global_home.latitude,
        global_home.longitude,
        global_home.altitude,
    )[:2]


def gps_to_local(
    gps: tuple[float, float], global_home: GlobalPosition
) -> tuple[float, float]:
    """
    converts gps coords to local coords.
    """

    return pm.geodetic2enu(
        *gps,
        global_home.altitude,
        global_home.latitude,
        global_home.longitude,
        global_home.altitude,
    )[:2]


def calculate_alignment_yaw(starting_position: np.ndarray, target_position: np.ndarray):
    # compute yaw
    unit = (target_position - starting_position) / np.linalg.norm(
        target_position - starting_position
    )
    dx = unit[0]
    dy = unit[1]
    yaw = np.arctan2(dx, dy)

    return yaw % (2 * np.pi)


if __name__ == "__main__":
    print(read_gpx_file("data/main_field.gpx"))
