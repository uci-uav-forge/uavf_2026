from dataclasses import dataclass
from .geofence import Geofence


class Djikstras:
    @dataclass
    class NodeInfo:
        shortest_dist: float | None
        shortest_node: tuple[float, float] | None

    def __init__(self, inset_points: list[tuple[float, float]], geofence: Geofence):
        self.inset_points = inset_points
        self.geofence = geofence
        self.d_table: dict[tuple[float, float], Djikstras.NodeInfo] = {}

        for point in self.inset_points:
            self.d_table[point] = Djikstras.NodeInfo(None, None)

    def get_path(
        self, start: tuple[float, float], end: tuple[float, float]
    ) -> list[tuple[float, float]]:
        self.start_node = start
        self.end_node = end
        self.d_table[start] = Djikstras.NodeInfo(0.0, None)
        self.d_table[end] = Djikstras.NodeInfo(None, None)
        self._populate_table()
        return self._shortest_path()

    @staticmethod
    def _dist(start: tuple[float, float], end: tuple[float, float]) -> float:
        return ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) ** 0.5

    def _populate_table(self):
        nodes = list(self.d_table.keys())
        node = self.start_node
        while True:
            past_dist = self.d_table[node].shortest_dist
            for neighbor in self.d_table:
                if neighbor is not node:
                    dist = Djikstras._dist(node, neighbor) + past_dist
                    if (not self.geofence.intersect(node, neighbor)) and (
                        self.d_table[neighbor].shortest_node is None
                        or dist < self.d_table[neighbor].shortest_dist
                    ):
                        self.d_table[neighbor] = Djikstras.NodeInfo(dist, node)
            nodes.remove(node)
            valid_nodes = list(
                filter(lambda n: self.d_table[n].shortest_dist is not None, nodes)
            )
            if len(valid_nodes) == 0:
                break
            else:
                node = valid_nodes[0]

    def _shortest_path(self) -> list[tuple[float, float]]:
        tmp = self.end_node
        path = []
        ctr = len(self.d_table)
        while tmp != self.start_node and ctr > 0:
            ctr -= 1
            path.append(tmp)
            tmp = self.d_table[tmp].shortest_node
        path.append(tmp)

        if ctr == 0:
            raise RuntimeError("Djikstras failed")

        path.reverse()
        return path
