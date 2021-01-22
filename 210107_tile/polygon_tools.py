from shapely import geometry


def if_in_poly(polygon, Points):
    line = geometry.LineString(polygon)
    point = geometry.Point(Points)
    polygon = geometry.Polygon(line)
    return polygon.contains(point)


if __name__ == "__main__":
    square = [(0, 0), (1, 0), (1, 1), (0, 1)]  # 多边形坐标
    pt1 = (2, 2)  # 点坐标
    pt2 = (0.5, 0.5)
    print(if_in_poly(square, pt1))
    print(if_in_poly(square, pt2))
