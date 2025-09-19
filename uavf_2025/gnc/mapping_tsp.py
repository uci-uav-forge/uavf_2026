import math
from pathlib import Path
from time import strftime

import cv2
import networkx as nx
import numpy as np
from PIL import Image
from line_profiler import profile

HEATMAP_DIM = 1000
FT_PER_METER = 3.28084


class MappingPathPlanner:
    """
    Generates mapping path.
    """

    def __init__(
        self,
        heatmap: np.ndarray,
        roi_corners: tuple[
            tuple[float, float], tuple[float, float], tuple[float, float]
        ],
        m_height: float,
        min_unmapped: float,
    ) -> None:
        """
        heatmap: Grayscale image where black pixels indicate unmapped areas
        roi_corners: Three corners of the ROI, considered as top left, top right, and bottom left corners during calculations
        min_unmapped: Value between 0 and 1 indicating minimum ratio of black pixels in a segment to be added as vertex of graph
        s: Side length of squares when segmenting ROI
        """
        self._heatmap = heatmap
        self._roi_corners = roi_corners
        self._m_height = m_height
        self._min_unmapped = min_unmapped
        self._ft_per_meter = FT_PER_METER

        self._G = nx.Graph()
        self._s = 0
        self._local_path = []
        self._initialize_graph_params()

    def _initialize_graph_params(self) -> None:
        """
        Calculate s value such that ROI is segmented into squares that are equal to or smaller
        than the area covered by the camera.
        """
        ft_height = self._m_height * self._ft_per_meter

        camera_y = 45 * ft_height / 50

        x_ft_per_pixel = (
            np.linalg.norm(np.subtract(self._roi_corners[0], self._roi_corners[1]))
            * self._ft_per_meter
        ) / HEATMAP_DIM
        y_ft_per_pixel = (
            np.linalg.norm(np.subtract(self._roi_corners[0], self._roi_corners[2]))
            * self._ft_per_meter
        ) / HEATMAP_DIM

        camera_length = camera_y / max(x_ft_per_pixel, y_ft_per_pixel)

        self._s = int(camera_length / math.sqrt(2))

    @profile
    def _create_vertices(self) -> None:
        """
        Vertices are in the format (x, y), which means that when represented in pixels
        the format is (column, row).
        """
        x_y_slices = self._divide_roi()
        for i, x in enumerate(x_y_slices):
            for j, y in enumerate(x_y_slices):
                row_start = x_y_slices[j - 1] if j > 0 else 0
                col_start = x_y_slices[i - 1] if i > 0 else 0
                total_px = (y - row_start) * (x - col_start)
                zero_count = total_px - np.count_nonzero(
                    self._heatmap[row_start:y, col_start:x]
                )
                if zero_count / total_px >= self._min_unmapped:
                    v_x = min(col_start + (self._s / 2), HEATMAP_DIM - 1)
                    v_y = min(row_start + (self._s / 2), HEATMAP_DIM - 1)
                    self._G.add_node(
                        (self._pixel_to_local((v_x, v_y))), heatmap_coord=(v_x, v_y)
                    )

    def _divide_roi(self):
        x_y_slices: list[int] = []
        x_y = self._s
        while x_y < HEATMAP_DIM:
            x_y_slices.append(x_y)
            x_y += self._s
        x_y_slices.append(HEATMAP_DIM)

        return x_y_slices

    @profile
    def _create_edges(self):
        V = list(self._G.nodes)
        total_V = len(V)
        for i in range(total_V - 1):
            for j in range(i + 1, total_V):
                self._G.add_edge(
                    V[i], V[j], weight=np.linalg.norm(np.subtract(V[i], V[j]))
                )

    def _generate_tsp_cycle(self) -> list[tuple[float, float]]:
        return nx.approximation.traveling_salesman_problem(
            self._G, method=nx.approximation.christofides
        )  # type: ignore

    def _cycle_to_path(
        self, current_pos: tuple[float, ...], cycle: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """
        Change Traveling Salesman Problem cycle to path starting at the vertex closest to the
        drone's current position.
        """
        start_idx = min(
            range(len(cycle)),
            key=lambda i: np.linalg.norm(np.subtract(current_pos[0:2], cycle[i])),
        )
        cycle.pop()
        return cycle[start_idx:] + cycle[0:start_idx]

    @profile
    def construct_path(
        self, current_pos: tuple[float, float] | tuple[float, float, float]
    ) -> list[tuple[float, float, float]]:
        self._create_vertices()
        if self._G.number_of_nodes() <= 1:
            self._local_path = list(self._G.nodes)
        else:
            self._create_edges()

            cycle = self._generate_tsp_cycle()
            self._local_path = self._cycle_to_path(current_pos, cycle)
        return list(
            map(
                lambda coord: (round(coord[0], 3), round(coord[1], 3), self._m_height),
                self._local_path,
            )
        )

    def _pixel_to_local(self, px_coord: tuple[float, float]) -> tuple[float, float]:
        """
        Pixel coordinates are in form (column, row) to match the (x, y) format of vertices.
        """
        top_left = np.array(self._roi_corners[0])
        horiz_dist = np.multiply(
            px_coord[0] / HEATMAP_DIM,
            np.subtract(self._roi_corners[1], self._roi_corners[0]),
        )
        vert_dist = np.multiply(
            px_coord[1] / HEATMAP_DIM,
            np.subtract(self._roi_corners[2], self._roi_corners[0]),
        )
        return tuple(top_left + horiz_dist + vert_dist)

    def _draw_vertices_on(self, img: np.ndarray) -> None:
        """
        Row and column where vertices show up may be off by 0.5.
        """
        vertices = list(self._G.nodes[v]["heatmap_coord"] for v in self._G.nodes)
        for v in vertices:
            v = tuple(int(x) for x in v)
            cv2.line(img, v, v, (0, 255, 0), 20)

    def _draw_tsp_edges_on(self, img: np.ndarray) -> None:
        if len(self._local_path) <= 1:
            return
        for i in range(len(self._local_path) - 1):
            u, v = (
                self._G.nodes[self._local_path[i]]["heatmap_coord"],
                self._G.nodes[self._local_path[i + 1]]["heatmap_coord"],
            )
            u, v = tuple(int(x) for x in u), tuple(int(x) for x in v)
            cv2.line(img, u, v, (0, 0, 255), 5)
            cv2.putText(img, f"{i}", u, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # Number last vertex in path
        cv2.putText(img, f"{i + 1}", v, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    def draw_path(self, img: np.ndarray) -> np.ndarray:
        img_dim = img.shape
        if len(img_dim) < 3 or img_dim[2] < 3:
            img = np.array(Image.fromarray(img).convert("RGB"))

        self._draw_vertices_on(img)
        self._draw_tsp_edges_on(img)
        return img

    def save_path_image(self, img: np.ndarray):
        time_string = strftime("%Y-%m-%d_%H-%M")
        mapping_logs_path = Path(__file__).absolute().parent.parent.parent / Path(
            f"logs/mapping_path/{time_string}"
        )
        mapping_logs_path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(mapping_logs_path / Path("planned_path.png")), self.draw_path(img)
        )


if __name__ == "__main__":
    # Example usage
    heatmap = np.zeros((HEATMAP_DIM, HEATMAP_DIM), dtype=np.uint8)
    roi_corners = ((0, 0), (HEATMAP_DIM, 0), (0, HEATMAP_DIM))
    m_height = 10.0
    min_unmapped = 0.1

    planner = MappingPathPlanner(heatmap, roi_corners, m_height, min_unmapped)
    current_position = (500, 500)  # Example current position
    path = planner.construct_path(current_position)

    import csv

    with open("mapping_path.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x", "y", "z"])
        writer.writerows(path)
