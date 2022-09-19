from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np
import torch.nn
from torchvision.models.detection import (
    RetinaNet_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
)

from detector_kit.entities import BoundingBox, Point

PREDICTION_BBOXES_KEY = "boxes"
PREDICTION_SCORES_KEY = "scores"
PREDICTION_LABELS_KEY = "labels"


class ObjectDetector:
    @classmethod
    def init_retina_net(cls, device: torch.device) -> ObjectDetector:
        weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
        model = retinanet_resnet50_fpn_v2(weights=weights).to(device)
        model.eval()
        return cls(
            model=model,
            pre_processing=partial(
                pre_process_object_detector_input, model_transforms=weights.transforms()
            ),
            post_processing=partial(
                post_process_retina_net_predictions,
                class_names=weights.meta["categories"],
            ),
            device=device,
        )

    def __init__(
        self,
        model: torch.nn.Module,
        pre_processing: Callable[[list[np.ndarray]], torch.Tensor],
        post_processing: Callable[[list[dict]], list[list[BoundingBox]]],
        device: torch.device,
    ):
        self.__model = model
        self.__pre_processing = pre_processing
        self.__post_processing = post_processing
        self.__device = device

    def detect_objects(
        self, image: np.ndarray, confidence_threshold: float
    ) -> list[BoundingBox]:
        batch = self.__pre_processing([image]).to(self.__device)
        with torch.no_grad():
            predictions = self.__model(batch)
        post_processed_predictions = self.__post_processing(predictions)[0]
        return filter_bounding_boxes(
            bounding_boxes=post_processed_predictions,
            confidence_threshold=confidence_threshold,
        )


def pre_process_object_detector_input(
    images: list[np.ndarray], model_transforms: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    raw_images = [torch.from_numpy(image).permute(2, 0, 1) for image in images]
    return torch.stack([model_transforms(image) for image in raw_images], dim=0)


def post_process_retina_net_predictions(
    raw_predictions: list[dict],
    class_names: list[str],
) -> list[list[BoundingBox]]:
    return [
        post_process_retina_net_prediction(
            prediction=prediction, class_names=class_names
        )
        for prediction in raw_predictions
    ]


def post_process_retina_net_prediction(
    prediction: dict, class_names: list[str]
) -> list[BoundingBox]:
    iterable = zip(
        prediction[PREDICTION_BBOXES_KEY],
        prediction[PREDICTION_SCORES_KEY],
        prediction[PREDICTION_LABELS_KEY],
    )
    result = []
    for raw_bbox, score, class_idx in iterable:
        bounding_box = create_bounding_box(
            raw_bbox=raw_bbox, score=score, label=class_names[class_idx.item()]
        )
        result.append(bounding_box)
    return result


def create_bounding_box(
    raw_bbox: torch.Tensor, score: torch.Tensor, label: str
) -> BoundingBox:
    left_top = Point(
        x=round_to_int(raw_bbox[0].item()), y=round_to_int(raw_bbox[1].item())
    )
    right_bottom = Point(
        x=round_to_int(raw_bbox[2].item()), y=round_to_int(raw_bbox[3].item())
    )
    return BoundingBox(
        left_top=left_top,
        right_bottom=right_bottom,
        confidence=score.item(),
        label=label,
    )


def round_to_int(value: float) -> int:
    return int(round(value, 0))


def filter_bounding_boxes(
    bounding_boxes: list[BoundingBox], confidence_threshold: float
) -> list[BoundingBox]:
    return [
        bounding_box
        for bounding_box in bounding_boxes
        if bounding_box.confidence >= confidence_threshold
    ]
