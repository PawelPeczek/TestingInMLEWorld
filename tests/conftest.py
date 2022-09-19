import os

import cv2
import numpy as np
import pytest

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
EXAMPLE_IMAGE_PATH = os.path.join(DATA_PATH, "example_image.jpg")


@pytest.fixture
def example_image() -> np.ndarray:
    return (cv2.imread(EXAMPLE_IMAGE_PATH)[:, :, ::-1]).copy()
