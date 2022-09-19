from detector_kit.entities import Point


def test_point_to_tuple() -> None:
    # given
    point = Point(x=37, y=90)

    # when
    result = point.to_tuple()

    # then
    assert result == (37, 90)
