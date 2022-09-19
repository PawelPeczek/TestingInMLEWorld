from unittest import mock
from unittest.mock import MagicMock, call

import numpy as np
import pytest
import torch

from detector_kit import detector
from detector_kit.detector import (
    filter_bounding_boxes,
    round_to_int,
    create_bounding_box,
    post_process_retina_net_prediction,
    post_process_retina_net_predictions,
    pre_process_object_detector_input,
    ObjectDetector,
)
from detector_kit.entities import BoundingBox, Point


def test_filter_bounding_boxes_when_empty_list_provided() -> None:
    # when
    result = filter_bounding_boxes(bounding_boxes=[], confidence_threshold=0.5)

    # then
    assert result == []


def test_filter_bounding_boxes_when_single_element_list_provided_and_threshold_is_met() -> None:
    # given
    bounding_boxes = [
        BoundingBox(
            confidence=0.51,
            label="some",
            left_top=MagicMock(),
            right_bottom=MagicMock(),
        )
    ]

    # when
    result = filter_bounding_boxes(
        bounding_boxes=bounding_boxes, confidence_threshold=0.5
    )

    # then
    assert result == bounding_boxes


def test_filter_bounding_boxes_when_single_element_list_provided_and_threshold_is_not_met() -> None:
    # given
    bounding_boxes = [
        BoundingBox(
            confidence=0.49,
            label="some",
            left_top=MagicMock(),
            right_bottom=MagicMock(),
        )
    ]

    # when
    result = filter_bounding_boxes(
        bounding_boxes=bounding_boxes, confidence_threshold=0.5
    )

    # then
    assert result == []


def test_filter_bounding_boxes_when_multiple_elements_list_provided_and_threshold_is_always_met() -> None:
    # given
    bounding_boxes = [
        BoundingBox(
            confidence=0.5, label="some", left_top=MagicMock(), right_bottom=MagicMock()
        ),
        BoundingBox(
            confidence=0.55,
            label="other",
            left_top=MagicMock(),
            right_bottom=MagicMock(),
        ),
    ]

    # when
    result = filter_bounding_boxes(
        bounding_boxes=bounding_boxes, confidence_threshold=0.5
    )

    # then
    assert result == bounding_boxes


def test_filter_bounding_boxes_when_multiple_elements_list_provided_and_threshold_is_never_met() -> None:
    # given
    bounding_boxes = [
        BoundingBox(
            confidence=0.51,
            label="some",
            left_top=MagicMock(),
            right_bottom=MagicMock(),
        ),
        BoundingBox(
            confidence=0.55,
            label="other",
            left_top=MagicMock(),
            right_bottom=MagicMock(),
        ),
    ]

    # when
    result = filter_bounding_boxes(
        bounding_boxes=bounding_boxes, confidence_threshold=0.56
    )

    # then
    assert result == []


def test_filter_bounding_boxes_when_multiple_elements_list_provided_and_threshold_is_sometimes_met() -> None:
    # given
    bounding_boxes = [
        BoundingBox(
            confidence=0.40,
            label="some",
            left_top=MagicMock(),
            right_bottom=MagicMock(),
        ),
        BoundingBox(
            confidence=0.5,
            label="other",
            left_top=MagicMock(),
            right_bottom=MagicMock(),
        ),
    ]

    # when
    result = filter_bounding_boxes(
        bounding_boxes=bounding_boxes, confidence_threshold=0.5
    )

    # then
    assert result == [bounding_boxes[1]]


def test_round_to_int_when_value_should_be_taken_as_ceil() -> None:
    # when
    result = round_to_int(value=3.5)

    # then
    assert result == 4


def test_round_to_int_when_value_should_be_taken_as_floor() -> None:
    # when
    result = round_to_int(value=3.4)

    # then
    assert result == 3


def test_create_bounding_box_when_input_is_valid() -> None:
    # given
    raw_bbox = torch.FloatTensor([33.3, 75.5, 193.2, 632.3])
    score = torch.FloatTensor([0.39])
    label = "some"

    # when
    result = create_bounding_box(raw_bbox=raw_bbox, score=score, label=label)

    # then
    assert result == BoundingBox(
        left_top=Point(x=33, y=76),
        right_bottom=Point(x=193, y=632),
        confidence=score.item(),
        label="some",
    )


def test_create_bounding_box_when_input_is_invalid() -> None:
    # given
    raw_bbox = torch.FloatTensor([33.3, 75.5])
    score = torch.FloatTensor([0.39])
    label = "some"

    # when
    with pytest.raises(IndexError):
        _ = create_bounding_box(raw_bbox=raw_bbox, score=score, label=label)


def test_post_process_retina_net_prediction_when_input_is_valid() -> None:
    # given
    prediction = {
        "boxes": torch.FloatTensor(
            [
                [33.3, 75.5, 193.2, 632.3],
                [43.3, 85.5, 293.2, 732.3],
            ]
        ),
        "scores": torch.FloatTensor([0.39, 0.49]),
        "labels": torch.IntTensor([1, 0]),
    }

    # when
    result = post_process_retina_net_prediction(
        prediction=prediction, class_names=["a", "b"]
    )

    # then
    assert result == [
        BoundingBox(
            left_top=Point(x=33, y=76),
            right_bottom=Point(x=193, y=632),
            confidence=prediction["scores"][0].item(),
            label="b",
        ),
        BoundingBox(
            left_top=Point(x=43, y=86),
            right_bottom=Point(x=293, y=732),
            confidence=prediction["scores"][1].item(),
            label="a",
        ),
    ]


def test_post_process_retina_net_prediction_when_prediction_is_invalid() -> None:
    # given
    prediction: dict[str, torch.Tensor] = {}

    # when
    with pytest.raises(KeyError):
        _ = post_process_retina_net_prediction(
            prediction=prediction, class_names=["some"]
        )


def test_post_process_retina_net_prediction_when_class_names_are_invalid() -> None:
    # given
    prediction = {
        "boxes": torch.FloatTensor([[33.3, 75.5, 193.2, 632.3]]),
        "scores": torch.FloatTensor([0.39]),
        "labels": torch.IntTensor([1]),
    }

    # when
    with pytest.raises(IndexError):
        _ = post_process_retina_net_prediction(prediction=prediction, class_names=[])


def test_post_process_retina_net_predictions_when_empty_input_is_given() -> None:
    # when
    result = post_process_retina_net_predictions(
        raw_predictions=[], class_names=["a", "b"]
    )

    # then
    assert result == []


@mock.patch.object(detector, "post_process_retina_net_prediction")
def test_post_process_retina_net_predictions_when_invalid_input_is_given(
    post_process_retina_net_prediction_mock: MagicMock,
) -> None:
    # given
    my_exception = KeyError()
    post_process_retina_net_prediction_mock.side_effect = my_exception

    # when
    with pytest.raises(KeyError) as error:
        _ = post_process_retina_net_predictions(
            raw_predictions=[{}, {}], class_names=["a", "b"]
        )

    # then
    assert error.value is my_exception


@mock.patch.object(detector, "post_process_retina_net_prediction")
def test_post_process_retina_net_predictions_when_valid_input_is_given(
    post_process_retina_net_prediction_mock: MagicMock,
) -> None:
    # given
    raw_predictions = [MagicMock(), MagicMock()]
    class_names = ["a", "b"]

    # when
    result = post_process_retina_net_predictions(
        raw_predictions=raw_predictions,  # type: ignore
        class_names=class_names,
    )

    # then
    post_process_retina_net_prediction_mock.assert_has_calls(
        [
            call(prediction=raw_predictions[0], class_names=class_names),
            call(prediction=raw_predictions[1], class_names=class_names),
        ]
    )
    assert result == [post_process_retina_net_prediction_mock.return_value] * 2


def test_pre_process_object_detector_input_when_empty_input_given() -> None:
    # when
    with pytest.raises(RuntimeError):
        _ = pre_process_object_detector_input(images=[], model_transforms=MagicMock())


def test_pre_process_object_detector_input_when_non_empty_input_given_with_the_same_size() -> None:
    # given
    images = [
        np.ones(shape=(720, 1280, 3), dtype=np.uint8),
        np.ones(shape=(720, 1280, 3), dtype=np.uint8),
    ]
    model_transforms = lambda x: x / 255

    # when
    result = pre_process_object_detector_input(
        images=images, model_transforms=model_transforms
    )

    # then
    assert result.shape == (2, 3, 720, 1280)
    assert ((result - 1 / 255) < 1e-5).all()


def test_pre_process_object_detector_input_when_non_empty_input_given_with_different_size() -> None:
    # given
    images = [
        np.ones(shape=(720, 1280, 3), dtype=np.uint8),
        np.ones(shape=(1080, 1920, 3), dtype=np.uint8),
    ]
    model_transforms = lambda x: x / 255

    # when
    with pytest.raises(RuntimeError):
        _ = pre_process_object_detector_input(
            images=images, model_transforms=model_transforms
        )


def test_object_detector_detect_objects_with_mocks() -> None:
    # given
    model = MagicMock()
    pre_processing = MagicMock()
    post_processing = MagicMock()
    expected_result = [
        BoundingBox(
            left_top=Point(x=33, y=76),
            right_bottom=Point(x=193, y=632),
            confidence=0.75,
            label="b",
        )
    ]
    post_processing.return_value = [expected_result]
    device = MagicMock()
    image = MagicMock()
    object_detector = ObjectDetector(
        model=model,
        pre_processing=pre_processing,
        post_processing=post_processing,
        device=device,
    )

    # when
    result = object_detector.detect_objects(image=image, confidence_threshold=0.4)

    # then
    pre_processing.assert_called_once_with([image])
    pre_processing.return_value.to.assert_called_once_with(device)
    model.assert_called_once_with(pre_processing.return_value.to.return_value)
    post_processing.assert_called_once_with(model.return_value)
    assert result == expected_result


def test_object_detector_detect_objects_end_to_end(example_image: np.ndarray) -> None:
    # given
    object_detector = ObjectDetector.init_retina_net(device=torch.device("cpu"))

    # when
    result = object_detector.detect_objects(
        image=example_image, confidence_threshold=0.4
    )

    # then
    assert len(result) > 0
