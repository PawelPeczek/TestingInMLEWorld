from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    x: int
    y: int

    def to_tuple(self) -> tuple[int, int]:
        return self.x, self.y


@dataclass(frozen=True)
class BoundingBox:
    label: str
    confidence: float
    left_top: Point
    right_bottom: Point
