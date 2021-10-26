def norm(point1: tuple, point2: tuple) -> int:
    """Calculate the norm of a vector using its two points

        :param point1: first point
        :param point2: second point
        :return: The calculated norm value
    """
    x, y = point1
    a, b = point2
    diff: tuple = tuple((x - a, y - b))  # return vector2D :points1 - points2 (u, v)
    u, v = diff
    return (u ** 2 + v ** 2) ** 0.5
