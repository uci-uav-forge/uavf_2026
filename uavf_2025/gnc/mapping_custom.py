import numpy as np
from scipy import ndimage
from pathlib import Path
from time import strftime

# For testing purposes only
import cv2
import matplotlib
import logging

import matplotlib.pyplot as plt
import random
from shapely.geometry import LineString, Point, Polygon

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
matplotlib.use("Agg")

IMAGE_DIM = 1000
X_FT_PER_PIXEL = 1.083
Y_FT_PER_PIXEL = 0.46


def remove_coord(
    coord_idx: int,
    col_diff: int,
    queue: list[list[tuple]],
    rectangles: list[tuple[tuple]],
    *,
    final_pass=False,
) -> int:
    """
    :params
    coord_idx: index of the rectangle in the queue to remove
    col_diff: maximum difference in columns spanned by two rectangles being joined
    queue: rectangles still being constructed
    rectangles: rectangles that are done being constructed
    final_pass: Boolean value indicating no more rectangles will be added to the queue

    Given the index of a rectangle to remove from the queue, join the rectangle with other rectangles located in consecutive
    rows that cover roughly the same columns.  If the rectangle can not be joined with any rectangles in the queue or no more
    rectangles will be added to the queue in the future, remove the rectangle from the queue and add it to the list of
    finished rectangles.  Otherwise, leave the rectangle in the queue.  Return the index within the queue of the next
    rectangle to consider.
    """
    updated = False
    i = coord_idx + 1
    while i < len(queue) and queue[i][0][0] - queue[coord_idx][1][0] <= 1:
        top_rect = queue[coord_idx]
        bottom_rect = queue[i]
        if (
            top_rect[1][0] + 1 == bottom_rect[0][0]
            and abs(top_rect[1][1] - bottom_rect[1][1]) <= col_diff
            and abs(top_rect[0][1] - bottom_rect[0][1]) <= col_diff
        ):
            queue[coord_idx] = [
                (top_rect[0][0], min(top_rect[0][1], bottom_rect[0][1])),
                (bottom_rect[1][0], max(top_rect[1][1], bottom_rect[1][1])),
            ]
            queue.pop(i)
            updated = True
        else:
            i += 1

    if not updated or final_pass:
        rectangles.append(tuple(queue[coord_idx]))
        queue.pop(coord_idx)
        return coord_idx

    return coord_idx + 1


def get_black_rectangles(img: np.array, col_diff: int) -> list[tuple[tuple]]:
    """
    Return a list of coordinates defining the vertices of rectangular unmapped areas, where col_diff
    determines the maximum variance allowed for combining two consecutive rectangles together.
    """
    # Group together 0s in the same row
    structure = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    horiz_unmapped_regions = ndimage.label(img[:, :] == 0, structure)[0]

    # Get slice objects for the indices of each group of 0s
    horiz_region_coords = ndimage.find_objects(horiz_unmapped_regions)

    # Create rectangles from the groups of 0s to represent the boundaries of each unmapped region
    rectangles = []
    queue = []
    for cur in horiz_region_coords:
        i = 0
        while i < len(queue):
            coord = queue[i]
            # When the current slice has an index more than 1 away from a coordinate, check if the rectangle described by that
            # coordinate has finished growing
            if cur[0].start > coord[1][0] + 1:
                i = remove_coord(i, col_diff, queue, rectangles)
                continue
            # If the current slice is in the same row as and begins in a nearby column to a coordinate, grow that coordinate's rectangle
            if (cur[0].start == coord[0][0]) and (
                cur[1].start - coord[1][1]
                <= min(cur[1].stop - cur[1].start - 1, coord[1][1] - coord[0][1])
            ):
                coord[1] = (coord[1][0], cur[1].stop - 1)
                break
            # If the current slice is in the next row and covers approximately the same columns as a coordinate,
            # grow that coordinate's rectangle
            elif (abs(cur[1].start - coord[0][1]) <= col_diff) and (
                abs(cur[1].stop - 1 - coord[1][1]) <= col_diff
            ):
                coord[0] = (coord[0][0], min(coord[0][1], cur[1].start))
                coord[1] = (coord[1][0] + 1, max(coord[1][1], cur[1].stop - 1))
                break
            i += 1
        else:
            # If the current slice can't be used to grow any coordinates, make a new coordinate for it
            queue.append(
                [(cur[0].start, cur[1].start), (cur[0].start, cur[1].stop - 1)]
            )

    i = 0
    while i < len(queue):
        i = remove_coord(i, col_diff, queue, rectangles, final_pass=True)

    return rectangles


def find_closest_corner(px_rect: list[tuple[tuple]], px_cur: tuple) -> tuple:
    """
    :params
    px_rect: pixels values of top left and bottom right corners of rectangles
    px_cur: pixel values for current location

    Return the closest corner of the rectangle in pixels.
    """
    corners = [
        px_rect[0],
        (px_rect[0][0], px_rect[1][1]),
        (px_rect[1][0], px_rect[0][1]),
        px_rect[1],
    ]

    corner_dist_map = dict(
        zip(
            corners,
            map(lambda corner: np.linalg.norm(np.subtract(corner, px_cur)), corners),
        )
    )

    return min(corner_dist_map.items(), key=lambda kv: kv[1])


def get_next_rectangle(
    px_rects: list[tuple[tuple]], px_cur: tuple, *, only_above=False
) -> tuple[tuple]:
    """
    :params
    px_rects: pixel coordinates of the top left and bottom right corners of rectangles
    px_cur: pixel coordinates of current position
    only_above: when true, only pick rectangles that are on the same or a smaller row number than the current position's row

    Remove and return the closest rectangle to the current position from px_rects.
    """
    search_rects = []
    if only_above:
        search_rects = [rect for rect in px_rects if rect[1][0] <= px_cur[0]]
    if not search_rects:
        search_rects = px_rects
        only_above = False

    rect_closest_corner_map = dict(
        zip(
            search_rects,
            map(lambda rect: find_closest_corner(rect, px_cur), search_rects),
        )
    )
    closest_idx = min(
        range(len(search_rects)),
        key=lambda i: rect_closest_corner_map[search_rects[i]][1],
    )
    closest_rect = search_rects[closest_idx]

    if only_above:
        px_rects.remove(search_rects[closest_idx])
    else:
        px_rects.pop(closest_idx)
    return closest_rect, rect_closest_corner_map[closest_rect][0]


def remove_overlap_midpoint(
    rect: tuple[tuple], prev: list[tuple], half_picture_x: float
) -> tuple[tuple]:
    """
    :params
    rect: next black rectangle that will be traversed
    prev: path through ROI that has been constructed so far
    half_picture_x: half the pixel distance covered horizontally by the camera

    Combine together waypoints occuring at the boundary between two adjacent black rectangles that would otherwise
    result in doubling back over the same area or stopping at an intermediary waypoint when following a straight trajectory.
    """
    flight_direction = find_flight_direction(rect)
    if len(prev) >= 2:
        same_dim = np.where(np.isclose(prev[-2], prev[-1], 0, 1e-5))
        # If traveling rowwise or columnwise through both rectangles and there is overlap between the area
        # that can be seen, the path at the boundary between the two rectangles can be combined
        if (
            same_dim[0].size > 0
            and same_dim[0][0] == 1 - flight_direction
            and (
                rect[0][1 - flight_direction]
                < prev[-1][1 - flight_direction]
                < rect[1][1 - flight_direction]
                or abs(
                    rect[0][1 - flight_direction]
                    + half_picture_x
                    - prev[-1][1 - flight_direction]
                )
                <= half_picture_x
                or abs(
                    rect[1][1 - flight_direction]
                    - half_picture_x
                    - prev[-1][1 - flight_direction]
                )
                <= half_picture_x
            )
        ):
            # Modify the last two waypoints in the path to account for the entire horizontal/vertical boundary
            # between the two rectangles
            new_waypoints = [[0, 0], [0, 0]]
            new_waypoints[0][1 - flight_direction] = new_waypoints[1][
                1 - flight_direction
            ] = prev[-1][1 - flight_direction]
            all_positions = [
                half_picture_x
                if pos[flight_direction] == 0
                else (999 - half_picture_x)
                if pos[flight_direction] == 999
                else pos[flight_direction]
                for pos in [*prev[-2:], *rect]
            ]
            new_waypoints[0][flight_direction] = min(all_positions)
            new_waypoints[1][flight_direction] = max(all_positions)
            if prev[-2][flight_direction] > prev[-1][flight_direction]:
                prev[-2:] = [tuple(waypoint) for waypoint in new_waypoints[::-1]]
            else:
                prev[-2:] = [tuple(waypoint) for waypoint in new_waypoints]

            # Shrink the next rectangle that will be traversed depending on which side has an overlap
            # with the end of the existing path
            new_rect = [list(corner) for corner in rect]
            if abs(
                rect[0][1 - flight_direction] - prev[-1][1 - flight_direction]
            ) < abs(rect[1][1 - flight_direction] - prev[-1][1 - flight_direction]):
                new_rect[0][1 - flight_direction] = (
                    prev[-1][1 - flight_direction] + half_picture_x
                )
            else:
                new_rect[1][1 - flight_direction] = (
                    prev[-1][1 - flight_direction] - half_picture_x
                )

            # Case where the entire rectangle can be seen by flying between the last two waypoints
            # of the existing path
            if new_rect[0][1 - flight_direction] >= new_rect[1][1 - flight_direction]:
                return ((), ())

            return (tuple(new_rect[0]), tuple(new_rect[1]))

    return rect


def remove_overlap_corner(
    rect: tuple[tuple], prev: list[tuple], closest_corner, half_picture_x: float
) -> tuple[tuple]:
    """
    :params
    rect: next black rectangle that will be traversed
    prev: path through ROI that has been constructed so far
    half_picture_x: half the pixel distance covered horizontally by the camera

    Combine together waypoints occuring at the boundary between two adjacent black rectangles that would otherwise
    result in doubling back over the same area or stopping at an intermediary waypoint when following a straight trajectory.
    """
    flight_direction = find_flight_direction(rect)
    if len(prev) >= 2:
        same_dim = np.where(np.isclose(prev[-2], prev[-1], 0, 1e-5))
        # If traveling rowwise or columnwise through both rectangles and there is overlap between the area
        # that can be seen, the path at the boundary between the two rectangles can be combined
        if (
            same_dim[0].size > 0
            and same_dim[0][0] == 1 - flight_direction
            and (
                rect[0][1 - flight_direction]
                < prev[-1][1 - flight_direction]
                < rect[1][1 - flight_direction]
                or abs(
                    rect[0][1 - flight_direction]
                    + half_picture_x
                    - prev[-1][1 - flight_direction]
                )
                <= half_picture_x
                or abs(
                    rect[1][1 - flight_direction]
                    - half_picture_x
                    - prev[-1][1 - flight_direction]
                )
                <= half_picture_x
            )
        ):
            # Modify the last two waypoints in the path to account for the entire horizontal/vertical boundary
            # between the two rectangles
            new_waypoints = [[0, 0], [0, 0]]
            new_waypoints[0][1 - flight_direction] = new_waypoints[1][
                1 - flight_direction
            ] = prev[-1][1 - flight_direction]
            all_positions = [
                half_picture_x
                if pos[flight_direction] == 0
                else (999 - half_picture_x)
                if pos[flight_direction] == 999
                else pos[flight_direction]
                for pos in [*prev[-2:], *rect]
            ]
            new_waypoints[0][flight_direction] = min(all_positions)
            new_waypoints[1][flight_direction] = max(all_positions)
            if prev[-2][flight_direction] > prev[-1][flight_direction]:
                prev[-2:] = [tuple(waypoint) for waypoint in new_waypoints[::-1]]
            else:
                prev[-2:] = [tuple(waypoint) for waypoint in new_waypoints]

            # Shrink the next rectangle that will be traversed depending on which side has an overlap
            # with the end of the existing path
            new_rect = [list(corner) for corner in rect]
            new_corner = [*closest_corner]
            if abs(
                rect[0][1 - flight_direction] - prev[-1][1 - flight_direction]
            ) < abs(rect[1][1 - flight_direction] - prev[-1][1 - flight_direction]):
                new_rect[0][1 - flight_direction] = (
                    prev[-1][1 - flight_direction] + half_picture_x
                )
                if (
                    closest_corner[1 - flight_direction]
                    == rect[0][1 - flight_direction]
                ):
                    new_corner[1 - flight_direction] = new_rect[0][1 - flight_direction]
            else:
                new_rect[1][1 - flight_direction] = (
                    prev[-1][1 - flight_direction] - half_picture_x
                )
                if (
                    closest_corner[1 - flight_direction]
                    == rect[1][1 - flight_direction]
                ):
                    new_corner[1 - flight_direction] = new_rect[1][1 - flight_direction]

            # Case where the entire rectangle can be seen by flying between the last two waypoints
            # of the existing path
            if new_rect[0][1 - flight_direction] >= new_rect[1][1 - flight_direction]:
                return ((), ()), ()

            return (tuple(new_rect[0]), tuple(new_rect[1])), tuple(new_corner)

    return rect, closest_corner


def update_position_midpoint(
    px_rects: list[tuple[tuple]],
    px_cur: tuple,
    path: list[tuple],
    half_picture_x: float,
    *,
    only_above=False,
) -> tuple[tuple, tuple[tuple]]:
    """
    :params
    px_rects: black rectangles in pixel coordinates that are not traversed by the current path
    px_cur: current position in pixel coordinates
    path: path through ROI that has been constructed so far
    half_picture_x: half the pixel distance covered horizontally by the camera
    only_above: when true, only pick rectangles that are on the same or a smaller row number than the current position's row

    Return the current position of the drone and next closest rectangle to traverse after modifying
    the path to account for overlap between adjacent rectangles.
    """
    closest_rect = ((), ())
    while closest_rect is not None and not closest_rect[0]:
        closest_rect = (
            find_closest_rectangle(px_rects, px_cur, only_above=only_above)
            if px_rects
            else None
        )
        only_above = False
        if path and closest_rect is not None:
            closest_rect = remove_overlap_midpoint(closest_rect, path, half_picture_x)
            px_cur = path[-1]

    return px_cur, closest_rect


def update_position_corner(
    px_rects: list[tuple[tuple]],
    px_cur: tuple,
    path: list[tuple],
    half_picture_x: float,
    *,
    only_above=False,
) -> tuple[tuple, tuple[tuple]]:
    """
    :params
    px_rects: black rectangles in pixel coordinates that are not traversed by the current path
    px_cur: current position in pixel coordinates
    path: path through ROI that has been constructed so far
    half_picture_x: half the pixel distance covered horizontally by the camera
    only_above: when true, only pick rectangles that are on the same or a smaller row number than the current position's row

    Return the current position of the drone and next closest rectangle to traverse after modifying
    the path to account for overlap between adjacent rectangles.
    """
    closest_rect = ((), ())
    while closest_rect is not None and not closest_rect[0]:
        closest_rect, closest_corner = (
            get_next_rectangle(px_rects, px_cur, only_above=only_above)
            if px_rects
            else (None, None)
        )
        only_above = False
        if path and closest_rect is not None:
            closest_rect, closest_corner = remove_overlap_corner(
                closest_rect, path, closest_corner, half_picture_x
            )
            px_cur = path[-1]

    return px_cur, closest_rect, closest_corner


def find_flight_direction(rect: tuple[tuple]) -> int:
    """
    Return the index for pixel coordinates that is the longest side of the rectangle, i.e.
    return 0 if the distance between the rows is greater and 1 if the distance between the
    columns is greater.
    """
    if rect[1][0] - rect[0][0] > rect[1][1] - rect[0][1]:
        return 0
    else:
        return 1


def find_start_corner(px_rect: list[tuple[tuple]], px_cur: tuple) -> tuple:
    """
    :params
    px_rect: pixels values of top left and bottom right corners of rectangles
    px_cur: pixel values for current location

    Return the closest corner of the rectangle in pixels.
    """
    corners = [
        px_rect[0],
        (px_rect[0][0], px_rect[1][1]),
        (px_rect[1][0], px_rect[0][1]),
        px_rect[1],
    ]
    closest_idx = min(
        range(4), key=lambda i: np.linalg.norm(np.subtract(corners[i], px_cur))
    )
    return corners[closest_idx]


def find_closest_rectangle(
    px_rects: list[tuple[tuple]], px_cur: tuple, *, only_above=False
) -> tuple[tuple]:
    """
    :params
    px_rects: pixel coordinates of the top left and bottom right corners of rectangles
    px_cur: pixel coordinates of current position
    only_above: when true, only pick rectangles that are on the same or a smaller row number than the current position's row

    Remove and return the closest rectangle to the current position from px_rects.
    """
    search_rects = []
    if only_above:
        search_rects = [rect for rect in px_rects if rect[1][0] <= px_cur[0]]
    if not search_rects:
        search_rects = px_rects
        only_above = False

    closest_idx = min(
        range(len(search_rects)),
        key=lambda i: np.linalg.norm(
            np.subtract(np.divide(np.add(*search_rects[i]), 2), px_cur)
        ),
    )

    if only_above:
        px_rects.remove(search_rects[closest_idx])
        return search_rects[closest_idx]
    else:
        return px_rects.pop(closest_idx)


def path_in_rectangle(
    start_corner: tuple,
    rect: tuple[tuple],
    half_picture_x: float,
    half_picture_y: float,
) -> list[tuple]:
    """
    :params:
    start_corner: pixel values for the corner of the rectangle which traversal will start from
    rect: pixel values for the top left and bottom right corners of the rectangle
    half_picture_x: half the pixel distance covered horizontally by the camera
    half_picture_y: half the pixel distance covered vertically by the camera

    Return a path in pixel values to take pictures of the entire rectangle.
    """
    # If the rectangle is a single pixel, throw it out
    if rect[0] == rect[1]:
        return []

    # Traverse the rectangle along the longer dimension
    if rect[1][0] - rect[0][0] > rect[1][1] - rect[0][1]:
        flight_direction = 0
    else:
        flight_direction = 1

    # Calculate the bounds to move back and forth between during traversal
    right_bottom_bound = max(rect[1][flight_direction] - half_picture_y, half_picture_y)
    left_top_bound = min(
        rect[0][flight_direction] + half_picture_y, 999 - half_picture_y
    )
    if right_bottom_bound < left_top_bound:
        left_top_bound, right_bottom_bound = right_bottom_bound, left_top_bound

    current_pos = start_corner
    fly_down_right = False
    path = []

    while current_pos == start_corner or (
        current_pos[1 - flight_direction] < rect[1][1 - flight_direction]
        and rect[0][1 - flight_direction] < current_pos[1 - flight_direction]
    ):
        updated_pos = [0, 0]
        # Case where the rectangle is a single row or single column
        if rect[0][1 - flight_direction] == rect[1][1 - flight_direction]:
            pos = current_pos[1 - flight_direction]
        # Case where traversing top to bottom or left to right, depending on longer dimension
        elif (
            fly_down_right
            or current_pos[1 - flight_direction] == rect[0][1 - flight_direction]
        ):
            pos = min(
                current_pos[1 - flight_direction] + half_picture_x, 999 - half_picture_x
            )
            fly_down_right = True
            updated_pos[1 - flight_direction] = pos + half_picture_x
        # Case where traversing bottom to top or right to left, depending on longer dimension
        else:
            pos = max(
                current_pos[1 - flight_direction] - half_picture_x, half_picture_x
            )
            updated_pos[1 - flight_direction] = pos - half_picture_x

        waypoint = [0, 0]
        waypoint[1 - flight_direction] = pos
        # Case where on the left or top of the rectangle
        if (
            current_pos[flight_direction] == left_top_bound
            or current_pos[flight_direction] == rect[0][flight_direction]
        ):
            waypoint[flight_direction] = left_top_bound
            path.append(tuple(waypoint))
            if left_top_bound != right_bottom_bound:
                waypoint[flight_direction] = right_bottom_bound
                path.append(tuple(waypoint))
        # Case where on the right or bottom of the rectangle
        else:
            waypoint[flight_direction] = right_bottom_bound
            path.append(tuple(waypoint))
            if left_top_bound != right_bottom_bound:
                waypoint[flight_direction] = left_top_bound
                path.append(tuple(waypoint))

        updated_pos[flight_direction] = path[-1][flight_direction]
        current_pos = tuple(updated_pos)

    return path


def create_path(
    px_rects: list[tuple[tuple]],
    cur_pos: tuple,
    camera_x: int | float,
    camera_y: int | float,
    roi_origin: tuple[float],
    roi_right: tuple[float],
    roi_down: tuple[float],
    *,
    version: int,
) -> list[tuple]:
    """
    :params
    px_rects: pixel coordinates of the top left and bottom right corners of rectangles
    cur_pos: pixel coordinates of current position
    camera_x: horizontal distance captured by the camera in feet
    camera_y: vertical distance captured by the camera in feet
    version: 1 is the original (traverse rectangles in order of closest midpoint)
             2 traverses rectangles in order of closest corner
             3 accounts for overlap between adjacent rectangles and uses midpoints to choose the next rectangle
             4 accounts for overlap between adjacent rectangles and uses closest corner to choose the next rectangle
    (If using a version that removes the overlap seems to still have a lot of overlap, it's most likely due to
    the next rectangle(s) being chosen not being the rectangles that have overlap with the end of the
    current path.)

    Return a path to take pictures of all the rectangles in pixel coordinates.
    """
    px_rects = px_rects[:]

    # Calculate the actual proportions for ROI from GPX file
    x_ft_per_pixel = (
        np.linalg.norm(np.subtract(roi_origin, roi_right)) * 3.28084
    ) / IMAGE_DIM
    y_ft_per_pixel = (
        np.linalg.norm(np.subtract(roi_origin, roi_down)) * 3.28084
    ) / IMAGE_DIM
    half_picture_x = camera_x / (x_ft_per_pixel * 2)
    half_picture_y = camera_y / (y_ft_per_pixel * 2)

    # If want to use proportions for SUAS ROI
    """
    half_picture_x = camera_x / (X_FT_PER_PIXEL * 2)
    half_picture_y = camera_y / (Y_FT_PER_PIXEL * 2)
    """
    path = []
    if version == 1:
        cur_rect = find_closest_rectangle(px_rects, cur_pos, only_above=True)
        while cur_rect is not None:
            start_corner = find_start_corner(cur_rect, cur_pos)
            path += path_in_rectangle(
                start_corner, cur_rect, half_picture_x, half_picture_y
            )
            if path:
                cur_pos = path[-1]
            cur_rect = find_closest_rectangle(px_rects, cur_pos) if px_rects else None
    elif version == 2:
        cur_rect, start_corner = get_next_rectangle(px_rects, cur_pos, only_above=True)
        while cur_rect is not None:
            path += path_in_rectangle(
                start_corner, cur_rect, half_picture_x, half_picture_y
            )

            # Remove duplicate coordinates if traversal of two consecutive rectangles only
            # shares one point, because one rectangle is traversed rowwise while the other
            # is traversed columnwise
            # Necessity is debatable due to uncommon occurrence and lack of noticeable problem
            # in sim from flying to a waypoint the drone is already at
            """
            if (path and next_rect_path and path[-1][0] == next_rect_path[0][0] 
            and path[-1][1] == next_rect_path[0][1]):
                path += next_rect_path[1:]
            else:
                path += next_rect_path
            """
            if path:
                cur_pos = path[-1]
            cur_rect, start_corner = (
                get_next_rectangle(px_rects, cur_pos) if px_rects else (None, None)
            )
    elif version == 3:
        cur_pos, cur_rect = update_position_midpoint(
            px_rects, cur_pos, path, half_picture_x, only_above=True
        )
        while cur_rect is not None:
            start_corner = find_start_corner(cur_rect, cur_pos)
            path += path_in_rectangle(
                start_corner, cur_rect, half_picture_x, half_picture_y
            )

            if path:
                cur_pos = path[-1]
            cur_pos, cur_rect = update_position_midpoint(
                px_rects, cur_pos, path, half_picture_x
            )
    elif version == 4:
        cur_pos, cur_rect, start_corner = update_position_corner(
            px_rects, cur_pos, path, half_picture_x, only_above=True
        )
        while cur_rect is not None:
            print("Adding point in path")
            path += path_in_rectangle(
                start_corner, cur_rect, half_picture_x, half_picture_y
            )
            if len(path) > 50:
                break
            if path:
                cur_pos = path[-1]
            cur_pos, cur_rect, start_corner = update_position_corner(
                px_rects, cur_pos, path, half_picture_x
            )

    print(f"Length of path: {len(path)}")
    return path


def pixel_to_local(
    px_coord: tuple[float],
    roi_origin: tuple[float],
    roi_right: tuple[float],
    roi_down: tuple[float],
) -> tuple[float]:
    """
    :params
    px_coord: pixel coordinate in the form (row, column)
    roi_origin: top left corner of ROI in local coordinate frame in the form (x, y)
    roi_right: top right corner of ROI in local coordinate frame in the form (x, y)
    roi_down: bottom left corner of ROI in local coordinate frame in the form (x, y)

    Return the local coordinates given a pixel coordinate in the format (row, column).
    """
    x = (
        roi_origin[0]
        + (px_coord[1] / IMAGE_DIM) * (roi_right[0] - roi_origin[0])
        + (px_coord[0] / IMAGE_DIM) * (roi_down[0] - roi_origin[0])
    )
    y = (
        roi_origin[1]
        + (px_coord[1] / IMAGE_DIM) * (roi_right[1] - roi_origin[1])
        + (px_coord[0] / IMAGE_DIM) * (roi_down[1] - roi_origin[1])
    )
    return (x, y)


def local_to_pixel(
    local_coord: tuple[float],
    roi_origin: tuple[float],
    roi_right: tuple[float],
    roi_down: tuple[float],
) -> tuple[float]:
    """
    :params
    local_coord: local coordinate in the form (x, y)
    roi_origin: top left corner of ROI in local coordinate frame in the form (x, y)
    roi_right: top right corner of ROI in local coordinate frame in the form (x, y)
    roi_down: bottom left corner of ROI in local coordinate frame in the form (x, y)

    Return the pixel coordinate in the form (row, column) given a local coordinate.
    """
    right_diff = np.subtract(roi_right, roi_origin)
    down_diff = np.subtract(roi_down, roi_origin)
    roi_x = local_coord[0] - roi_origin[0]
    roi_y = roi_origin[1] - local_coord[1]
    denom = right_diff[0] * down_diff[1] - right_diff[1] * down_diff[0]

    px_row = ((right_diff[1] * roi_x + right_diff[0] * roi_y) / -denom) * IMAGE_DIM
    px_col = ((down_diff[1] * roi_x + down_diff[0] * roi_y) / denom) * IMAGE_DIM

    return (px_row, px_col)


def get_mapping_path(
    img: np.array,
    cur_local_pos: tuple,
    m_height: float,
    roi_origin: tuple[float],
    roi_right: tuple[float],
    roi_down: tuple[float],
) -> list[tuple[float]]:
    """
    :params
    img: numpy array representing a picture of how much of the ROI has been mapped
    cur_local_pos: starting position of the drone for the mapping task in the form (x, y, z)
    m_height: height that drone will fly at to take pictures for mapping in meters
    roi_origin: top left corner of ROI in local coordinate frame in the form (x, y)
    roi_right: top right corner of ROI in local coordinate frame in the form (x, y)
    roi_down: bottom left corner of ROI in local coordinate frame in the form (x, y)

    Return a path of (x, y, z) coordinates to take pictures of all unmapped regions in the ROI.
    """
    # Convert height from meters to feet
    ft_height = m_height * 3.28084

    # Calculate dimensions of the image the camera will capture
    camera_x = 85 * ft_height / 50
    camera_y = 45 * ft_height / 50

    # Find black rectangles in the provided image
    unmapped_rectangles = get_black_rectangles(img, 120)

    # Convert the drone's current local position to pixel coordinates
    cur_px_pos = local_to_pixel(cur_local_pos[0:2], roi_origin, roi_right, roi_down)

    # Create mapping path in pixel coordinates
    if not unmapped_rectangles:
        return []
    px_path = create_path(
        unmapped_rectangles,
        cur_px_pos,
        camera_x,
        camera_y,
        roi_origin,
        roi_right,
        roi_down,
        version=4,
    )

    # Convert the mapping path from pixel coordinates to local coordinates on xyz plane
    local_path = [0] * len(px_path)
    for i in range(len(px_path)):
        x, y = pixel_to_local(px_path[i], roi_origin, roi_right, roi_down)
        local_path[i] = (round(x, 3), round(y, 3), m_height)

    # Log coverage image and a graph of the planned path
    mapping_logs_path = create_log_folder()
    visualize_path(unmapped_rectangles, px_path, mapping_logs_path)
    cv2.imwrite(str(mapping_logs_path / Path("coverage_image.png")), img)

    return local_path


def visualize_path(unmapped_rectangles, pixel_path, mapping_logs_path: Path):
    if len(pixel_path) != 1:
        planned_path = LineString([[c, r] for r, c in pixel_path])
    else:
        planned_path = Point(pixel_path[0][0], pixel_path[0][1])

    plt.clf()

    for i, rect in enumerate(unmapped_rectangles):
        bottom_left, top_right = rect
        bottom_left_y, bottom_left_x = bottom_left
        top_right_y, top_right_x = top_right

        rectangle = Polygon(
            [
                (bottom_left_x, bottom_left_y),
                (bottom_left_x, top_right_y),
                (top_right_x, top_right_y),
                (top_right_x, bottom_left_y),
            ]
        )
        x, y = rectangle.exterior.xy
        plt.fill(x, y, color="lightskyblue", label=f"{i}")

    x, y = planned_path.xy
    plt.plot(x, y, marker="o")

    for i, txt in enumerate(y):
        plt.text(x[i], y[i], str(i + 1), ha="center", va="bottom", color="black")

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Planned Path")

    plt.savefig(mapping_logs_path / Path("planned_path.png"))


def create_log_folder() -> Path:
    """
    Create folder in uavf_2025/logs/mapping_path to store the coverage image and
    planned path for mapping.  Return a Path object for the directory.
    """
    time_string = strftime("%Y-%m-%d_%H-%M")
    mapping_logs_path = Path(__file__).absolute().parent.parent.parent / Path(
        f"logs/mapping_path/{time_string}"
    )
    mapping_logs_path.mkdir(parents=True, exist_ok=True)
    return mapping_logs_path


# For testing purposes only
def create_mock_image(
    px: int,
    num_rects: int = 0,
    max_x_step: int = 0,
    max_y_step: int = 0,
    from_corners: list[tuple[tuple]] = [],
    display: bool = True,
) -> np.array:
    """
    :params
    px: length/width of square image
    num_rects: number of gray rectangles to draw on the image
    max_x_step: largest length (along x-axis) possible for gray rectangles
    max_y_step: largest width (along y-axis) possible for gray rectangles
    from_corners: list of the top left and bottom right corners of gray rectangles drawn to
    create an image
    display: Boolean value indicating whether to show the image or not
    """
    img = np.zeros((px, px), dtype=np.uint8)
    if from_corners:
        for corner in from_corners:
            cv2.rectangle(img, corner[0], corner[1], 20, -1)
    else:
        assert (
            max_x_step != 0 and max_y_step != 0
        ), "ERROR: max_x_step and max_y_step must be greater than 0"
        corners = []
        for _ in range(num_rects):
            x1 = random.randrange(1, px)
            y1 = random.randrange(1, px)
            x2 = random.randrange(x1 + 1, min(x1 + max_x_step, px + 1))
            y2 = random.randrange(y1 + 1, min(y1 + max_y_step, px + 1))
            corners.append(((x1, y1), (x2, y2)))
            cv2.rectangle(img, (x1, y1), (x2, y2), 20, -1)
        print(f"Rectangle corners to recreate randomly generated image: {corners}")

    if display:
        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img


if __name__ == "__main__":
    img = create_mock_image(1000, 5, 500, 600, display=False)
    # corners = [((329, 788), (646, 811)), ((119, 207), (493, 382)), ((579, 283), (580, 376))]
    path = get_mapping_path(img, (63, 30), 50, (1, 1), (150, 5), (-1, 45))
    print(path)
