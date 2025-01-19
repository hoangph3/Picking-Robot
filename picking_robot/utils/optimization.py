from picking_robot.utils.calculation import calculate_centroid, calculate_iou
from picking_robot.utils.processing import rotate_box, resize_box
from tqdm import tqdm
import numpy as np


def find_similar_box(contour, bbox, widths=np.arange(50, 300, 10), angles=np.arange(-90, 90, 3), length_devide_by_width=1.82):
    center = calculate_centroid(contour)
    best_box = bbox
    best_iou = calculate_iou(contour, best_box)

    for width in tqdm(widths):  # brute-force width
        for angle in angles:  # brute-force angle
            resized_box = resize_box(center, width, width * length_devide_by_width)
            rotated_box = rotate_box(resized_box, angle, center)
            new_iou = calculate_iou(contour, rotated_box)
            if new_iou > best_iou:
                best_iou = new_iou
                best_box = rotated_box

    return best_box
