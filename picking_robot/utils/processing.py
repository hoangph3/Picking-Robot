import cv2
import numpy as np

from picking_robot.utils.calculation import calculate_distance


def rotate_image(image, rot_matrix):
    (h, w) = image.shape[:2]
    rotated_image = cv2.warpAffine(image, rot_matrix, (w, h))
    return rotated_image


def rotate_box(box, angle, center):
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_box = cv2.transform(np.array([box]), rot_matrix)[0]
    return rotated_box


def resize_box(center, width, length):
    box_points = np.array([
        [center[0] - width / 2, center[1] - length / 2],
        [center[0] + width / 2, center[1] - length / 2],
        [center[0] + width / 2, center[1] + length / 2],
        [center[0] - width / 2, center[1] + length / 2]], dtype=np.float32)
    return box_points


def translation_line(p1, p2, length):
    # vector perpendicular to current side
    vec = np.array([p2[1] - p1[1], p1[0] - p2[0]])
    vec_norm = vec / np.linalg.norm(vec)
    p1_new = p1 + vec_norm * length
    p2_new = p2 + vec_norm * length
    return p1_new, p2_new


def pad_rectangle_width(polygon_points, padding_rate):
    # Calculate the length of each side
    distances = [calculate_distance(polygon_points[i], polygon_points[(i + 1) % 4]) for i in range(4)]
    
    # Identify shorter sides (assumed to be opposite sides in a rectangle)
    if distances[0] < distances[1]:
        padding_width = distances[0] * padding_rate
        short_side_indices = [(0, 1), (2, 3)]
        long_side_indices = [(1, 2), (3, 0)]
    else:
        padding_width = distances[1] * padding_rate
        short_side_indices = [(1, 2), (3, 0)]
        long_side_indices = [(0, 1), (2, 3)]

    # Create a copy to store new points
    new_polygon_points = polygon_points.copy()

    for idx1, idx2 in long_side_indices:
        new_polygon_points[idx1], new_polygon_points[idx2] = translation_line(polygon_points[idx1], polygon_points[idx2], padding_width / 2)

    return new_polygon_points


def crop_image(image, box, x_pad=0, y_pad=0):
    (h, w) = image.shape[:2]
    xs = box[:,0]
    ys = box[:,1]
    xmin = np.min(xs).astype(np.int32)
    xmax = np.max(xs).astype(np.int32)
    ymin = np.min(ys).astype(np.int32)
    ymax = np.max(ys).astype(np.int32)

    x1 = max(xmin - x_pad, 0)
    x2 = min(xmax + x_pad, w)
    y1 = max(ymin - y_pad, 0)
    y2 = min(ymax + y_pad, h)

    cropped_image = image[y1:y2, x1:x2]
    return cropped_image


def get_width_and_length_points(points):
    pairwise_distances = {(i, (i + 1) % 4): calculate_distance(points[i], points[(i + 1) % 4]) for i in range(4)}
    sorted_distances = sorted(pairwise_distances.items(), key=lambda item: item[1])
    width_pairs = [(points[i], points[j]) for (i, j) in [pair[0] for pair in sorted_distances[:2]]]
    length_pairs = [(points[i], points[j]) for (i, j) in [pair[0] for pair in sorted_distances[2:]]]
    return width_pairs, length_pairs
