import numpy as np
from shapely.geometry import Polygon


def calculate_centroid(vertices):
    if isinstance(vertices, np.ndarray):
        vertices = vertices.tolist()

    n = len(vertices)
    if vertices[0] != vertices[-1]:
        vertices.append(vertices[0])  # Ensure the polygon is closed

    A = 0  # Signed area
    C_x = 0
    C_y = 0

    for i in range(n):
        x_i, y_i = vertices[i]
        x_next, y_next = vertices[i + 1]
        cross_product = x_i * y_next - x_next * y_i
        A += cross_product
        C_x += (x_i + x_next) * cross_product
        C_y += (y_i + y_next) * cross_product

    A *= 0.5
    C_x /= (6 * A)
    C_y /= (6 * A)

    return (C_x, C_y)


def calculate_iou(ref, target):
    poly_target = Polygon(target)
    poly_ref = Polygon(ref)
    intersect = poly_target.intersection(poly_ref).area
    union = poly_target.union(poly_ref).area
    iou = intersect / union
    return iou


def calculate_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    distance = (dx**2 + dy**2) ** 0.5
    return distance


def calculate_midpoint(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    midpoint_x = (x1 + x2) / 2
    midpoint_y = (y1 + y2) / 2
    return (midpoint_x, midpoint_y)


def angle_with_x_axis(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    dx = x2 - x1
    dy = y2 - y1
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    return angle_deg
