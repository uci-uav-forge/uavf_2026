from shapely.geometry import Polygon, MultiPolygon, LineString, Point


class Geofence:
    def __init__(self, verticies: list[tuple[float, float]]):
        self.polygon = Polygon(verticies)
        self.verticies = verticies

    def shrink(self, dist):
        """Returns a shrinked version of this polygon by dist units from the sides."""
        shrunken_polygon = self.polygon.buffer(-dist, join_style=2)
        print(f"shrunken_polygon: {shrunken_polygon}")
        if type(shrunken_polygon) is MultiPolygon:
            print("breaking up multipolygon")
            polygons = list(shrunken_polygon.geoms)
            shrunken_polygon = max(polygons, key=lambda poly: poly.area)

        return Geofence(list(shrunken_polygon.exterior.coords))  # type: ignore

    def get_coords(self) -> list[tuple[float, float]]:
        return list(self.polygon.exterior.coords)  # type: ignore

    def intersect(
        self, point1: tuple[float, float], point2: tuple[float, float]
    ) -> bool:
        line = LineString([point1, point2])
        return line.intersects(self.polygon.boundary)

    def contains(self, p: tuple[float, float]) -> bool:
        point = Point(*p)
        return self.polygon.contains(point)
