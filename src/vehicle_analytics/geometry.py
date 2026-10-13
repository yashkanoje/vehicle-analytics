"""Pure image-plane and ground-plane geometry. numpy and the standard library only."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

Point = tuple[float, float]


def reference_point(box_xyxy: Sequence[float]) -> Point:
    """Bottom-centre of a box: the point where the vehicle meets the road."""
    x1, _, x2, y2 = box_xyxy
    return ((x1 + x2) / 2.0, float(y2))


@dataclass(frozen=True)
class Line:
    """A named segment from p1 to p2, in pixels."""

    name: str
    p1: Point
    p2: Point

    def __post_init__(self) -> None:
        if self.p1 == self.p2:
            raise ValueError(f"line '{self.name}' has zero length")

    def signed_distance(self, p: Point) -> float:
        """Perpendicular distance from p to the infinite line through the segment.

        Positive on the side the vector (-dy, dx) points to. For a left-to-right
        horizontal line that is below the line, where y is larger.
        """
        dx = self.p2[0] - self.p1[0]
        dy = self.p2[1] - self.p1[1]
        cross = (p[1] - self.p1[1]) * dx - (p[0] - self.p1[0]) * dy
        return cross / math.hypot(dx, dy)

    def within_segment(self, p: Point) -> bool:
        """True when p projects onto the segment itself, not its extension."""
        dx = self.p2[0] - self.p1[0]
        dy = self.p2[1] - self.p1[1]
        u = ((p[0] - self.p1[0]) * dx + (p[1] - self.p1[1]) * dy) / (dx * dx + dy * dy)
        return 0.0 <= u <= 1.0


def _has_collinear_triple(points: Sequence[Point]) -> bool:
    pts = np.asarray(points, dtype=float)
    scale = float(np.ptp(pts, axis=0).max())
    if scale == 0.0:
        return True
    for i, j, k in itertools.combinations(range(len(pts)), 3):
        cross = (pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1]) - (pts[j, 1] - pts[i, 1]) * (
            pts[k, 0] - pts[i, 0]
        )
        if abs(cross) < 1e-9 * scale * scale:
            return True
    return False


@dataclass(frozen=True, eq=False)
class Homography:
    """Maps image points to ground-plane coordinates in metres."""

    matrix: np.ndarray

    @classmethod
    def from_points(
        cls, image_points: Sequence[Point], ground_points: Sequence[Point]
    ) -> Homography:
        if len(image_points) != 4 or len(ground_points) != 4:
            raise ValueError(
                "calibration needs exactly four image points and four ground points"
            )
        if _has_collinear_triple(image_points) or _has_collinear_triple(ground_points):
            raise ValueError("calibration points are degenerate (collinear or repeated)")
        rows: list[list[float]] = []
        rhs: list[float] = []
        for (x, y), (u, v) in zip(image_points, ground_points, strict=True):
            rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
            rhs.append(u)
            rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
            rhs.append(v)
        h = np.linalg.solve(np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float))
        return cls(np.append(h, 1.0).reshape(3, 3))

    def to_ground(self, p: Point) -> Point:
        x, y, w = self.matrix @ np.array([p[0], p[1], 1.0])
        return (float(x / w), float(y / w))
