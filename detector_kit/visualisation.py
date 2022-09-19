import numpy as np
import cv2

from detector_kit.entities import BoundingBox


def draw_bounding_boxes(
    image: np.ndarray, bounding_boxes: list[BoundingBox]
) -> np.ndarray:
    image_copy = image.copy()
    for bounding_box in bounding_boxes:
        image_copy = cv2.rectangle(
            image_copy,
            bounding_box.left_top.to_tuple(),
            bounding_box.right_bottom.to_tuple(),
            (0, 0, 255),
            2,
        )
    return image_copy
