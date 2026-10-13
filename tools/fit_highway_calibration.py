"""Reproduce the ground-plane calibration in configs/highway.yaml.

The coordinates below are pixel positions (640x360) of road markings, measured on a
vehicle-free median image of the development clip. The footage is third-party and is not
in this repository; these measurements are the evidence the calibration rests on.

Model: every lane-line mark lies on one ground plane. On the right carriageway the two
lane lines are at x = 0 and x = 3.70 m and the marks repeat every 9 m. The left
carriageway has the same pattern, shifted sideways by x0 and along the road by y0, both
unknown. The fit finds the ground-to-image homography, x0 and y0 that minimise the
reprojection error of all 21 marks.

Run:  python tools/fit_highway_calibration.py      (needs scipy)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from vehicle_analytics.geometry import Homography, Point

WIDTH, HEIGHT = 640, 360
LANE_M = 3.70  # middle lane of a dual three-lane motorway, DMRB CD 127 Figure 2.1.1N1a
MODULE_M = 9.0  # 2 m mark + 7 m gap, Traffic Signs Manual Chapter 5 Figure 2-3, diagram 1005.1
ZONE_MODULES = 4

# Centroids of lane-line marks, nearest first.
RIGHT_INNER = [  # between the middle and offside lanes of the right carriageway (x = 0)
    (437.98, 271.11), (408.32, 232.46), (390.56, 208.96),
    (378.42, 192.86), (370.21, 181.52), (364.23, 172.86),
]
RIGHT_OUTER = [  # between the nearside and middle lanes of the right carriageway (x = 3.70)
    (504.54, 271.07), (461.30, 234.36), (433.04, 210.23), (414.01, 193.95), (400.98, 182.63),
]
LEFT_OUTER = [  # between the nearside and middle lanes of the left carriageway (x = x0 - 3.70)
    (133.71, 266.31), (176.71, 230.09), (204.40, 206.78), (224.04, 190.32), (237.24, 179.36),
]
LEFT_INNER = [  # between the middle and offside lanes of the left carriageway (x = x0)
    (202.02, 263.66), (229.15, 227.88), (246.34, 205.78), (258.87, 189.34), (267.11, 178.69),
]
LANE_LINES = {
    "right inner": RIGHT_INNER,
    "right outer": RIGHT_OUTER,
    "left outer": LEFT_OUTER,
    "left inner": LEFT_INNER,
}

# Centres of the continuous edge lines at four places along the road. Not used by the fit:
# each should lie one outer lane (3.65 m) plus about half an edge line from its lane line.
EDGE_LINES = {  # edge line: (its neighbouring lane line, edge-line centres)
    "right nearside": (
        "right outer",
        [(535.27, 249.25), (488.82, 219.77), (458.92, 200.67), (438.06, 187.28)],
    ),
    "right offside": (
        "right inner",
        [(360.46, 250.11), (350.93, 219.91), (345.11, 200.50), (341.05, 186.97)],
    ),
    "left nearside": (
        "left outer",
        [(88.56, 251.44), (139.34, 220.13), (172.35, 200.24), (195.21, 186.50)],
    ),
    "left offside": (
        "left inner",
        [(277.32, 250.52), (286.19, 219.98), (291.65, 200.42), (295.95, 186.82)],
    ),
}


@dataclass(frozen=True)
class FitResult:
    rms_px: float
    max_px: float
    x0_m: float
    y0_m: float
    image_points: list[Point]  # fractions of the frame, as in the config
    ground_points_m: list[Point]


@dataclass(frozen=True)
class Checks:
    zone_length_m: dict[str, float]
    edge_distance_m: dict[str, list[float]]


def _ground_model(x0: float, y0: float) -> np.ndarray:
    points = [(0.0, MODULE_M * k) for k in range(len(RIGHT_INNER))]
    points += [(LANE_M, MODULE_M * k) for k in range(len(RIGHT_OUTER))]
    points += [(x0 - LANE_M, y0 + MODULE_M * k) for k in range(len(LEFT_OUTER))]
    points += [(x0, y0 + MODULE_M * k) for k in range(len(LEFT_INNER))]
    return np.array(points)


def _project(ground_to_image: np.ndarray, ground_points: np.ndarray) -> np.ndarray:
    matrix = np.append(ground_to_image, 1.0).reshape(3, 3)
    projected = np.c_[ground_points, np.ones(len(ground_points))] @ matrix.T
    return projected[:, :2] / projected[:, 2:3]


def fit() -> FitResult:
    measured = np.array(RIGHT_INNER + RIGHT_OUTER + LEFT_OUTER + LEFT_INNER)

    def residuals(params: np.ndarray) -> np.ndarray:
        return (_project(params[:8], _ground_model(params[8], params[9])) - measured).ravel()

    # Start from the middle lane of the right carriageway alone.
    zone_m = MODULE_M * ZONE_MODULES
    start = Homography.from_points(
        [RIGHT_INNER[0], RIGHT_OUTER[0], RIGHT_OUTER[ZONE_MODULES], RIGHT_INNER[ZONE_MODULES]],
        [(0.0, 0.0), (LANE_M, 0.0), (LANE_M, zone_m), (0.0, zone_m)],
    )
    ground_to_image = np.linalg.inv(start.matrix)
    ground_to_image /= ground_to_image[2, 2]
    solution = least_squares(
        residuals, np.r_[ground_to_image.ravel()[:8], -11.8, 0.8], x_scale="jac"
    )
    params = solution.x
    errors = np.hypot(*residuals(params).reshape(-1, 2).T)

    # Four well-spread pairs from the fitted plane: a rectangle across both carriageways.
    x0, y0 = float(params[8]), float(params[9])
    corners = [(x0 - LANE_M, 0.0), (LANE_M, 0.0), (LANE_M, zone_m), (x0 - LANE_M, zone_m)]
    pixels = _project(params[:8], np.array(corners))
    return FitResult(
        rms_px=float(np.sqrt(np.mean(errors**2))),
        max_px=float(errors.max()),
        x0_m=x0,
        y0_m=y0,
        image_points=[(round(float(x) / WIDTH, 4), round(float(y) / HEIGHT, 4)) for x, y in pixels],
        ground_points_m=[(round(x, 2), round(y, 2)) for x, y in corners],
    )


def check(homography: Homography) -> Checks:
    """Measure the marks with a calibration: zone length per lane line, lane line to edge line."""
    zone_length = {}
    line_x = {}
    for name, marks in LANE_LINES.items():
        ground = [homography.to_ground(p) for p in marks]
        steps = zip(ground[:ZONE_MODULES], ground[1 : ZONE_MODULES + 1], strict=True)
        zone_length[name] = sum(math.dist(a, b) for a, b in steps)
        line_x[name] = float(np.mean([x for x, _ in ground]))
    edge_distance = {
        name: [abs(homography.to_ground(p)[0] - line_x[lane_line]) for p in points]
        for name, (lane_line, points) in EDGE_LINES.items()
    }
    return Checks(zone_length_m=zone_length, edge_distance_m=edge_distance)


OUTER_LANE_M = 3.65  # DMRB CD 127 Figure 2.1.1N1a
SHOULDER_M = 3.30
ZONE_LENGTH_M = 40.0


def to_image(homography: Homography, ground: Point) -> Point:
    """Project a ground point (metres) back into the image (pixels)."""
    x, y, w = np.linalg.inv(homography.matrix) @ np.array([ground[0], ground[1], 1.0])
    return (float(x / w), float(y / w))


def lane_zones(result: FitResult) -> list[dict]:
    """Lanes and hard shoulders as image polygons, from their widths on the fitted plane."""
    pixels = [(x * WIDTH, y * HEIGHT) for x, y in result.image_points]
    homography = Homography.from_points(pixels, result.ground_points_m)
    x0 = result.x0_m
    bands = [
        ("down_lane_3", "lane", -OUTER_LANE_M, 0.0, "down"),
        ("down_lane_2", "lane", 0.0, LANE_M, "down"),
        ("down_lane_1", "lane", LANE_M, LANE_M + OUTER_LANE_M, "down"),
        (
            "down_shoulder",
            "shoulder",
            LANE_M + OUTER_LANE_M,
            LANE_M + OUTER_LANE_M + SHOULDER_M,
            None,
        ),
        ("up_lane_3", "lane", x0, x0 + OUTER_LANE_M, "up"),
        ("up_lane_2", "lane", x0 - LANE_M, x0, "up"),
        ("up_lane_1", "lane", x0 - LANE_M - OUTER_LANE_M, x0 - LANE_M, "up"),
        (
            "up_shoulder",
            "shoulder",
            x0 - LANE_M - OUTER_LANE_M - SHOULDER_M,
            x0 - LANE_M - OUTER_LANE_M,
            None,
        ),
    ]
    zones = []
    for name, kind, x_from, x_to, direction in bands:
        corners = [(x_from, 0.0), (x_to, 0.0), (x_to, ZONE_LENGTH_M), (x_from, ZONE_LENGTH_M)]
        polygon = []
        for corner in corners:
            x, y = to_image(homography, corner)
            # A corner that falls outside the frame is moved to the frame edge. That
            # narrows the zone there instead of cutting it: `up_shoulder` is about
            # 2.4 m wide at its near end, not 3.30 m.
            polygon.append(
                [round(min(max(x / WIDTH, 0.0), 1.0), 4), round(min(max(y / HEIGHT, 0.0), 1.0), 4)]
            )
        zone = {"name": name, "kind": kind, "polygon": polygon}
        if direction is not None:
            zone["allowed_direction"] = direction
        zones.append(zone)
    return zones


def print_zones(result: FitResult) -> None:
    print("zones:")
    for zone in lane_zones(result):
        print(f"  - name: {zone['name']}")
        print(f"    kind: {zone['kind']}")
        print(f"    polygon: {zone['polygon']}")
        if "allowed_direction" in zone:
            print(f"    allowed_direction: {zone['allowed_direction']}")


def main() -> None:
    result = fit()
    print(f"reprojection error: {result.rms_px:.2f} px RMS, {result.max_px:.2f} px max")
    print(f"left carriageway offset: x0 = {result.x0_m:.2f} m, y0 = {result.y0_m:.2f} m")
    print("calibration:")
    print("  image_points:")
    for x, y in result.image_points:
        print(f"    - [{x:.4f}, {y:.4f}]")
    print("  ground_points_m:")
    for x, y in result.ground_points_m:
        print(f"    - [{x:.2f}, {y:.2f}]")

    pixels = [(x * WIDTH, y * HEIGHT) for x, y in result.image_points]
    checks = check(Homography.from_points(pixels, result.ground_points_m))
    print(f"zone length over {ZONE_MODULES} modules (expected {MODULE_M * ZONE_MODULES:.1f} m):")
    for name, length in checks.zone_length_m.items():
        print(f"  {name:12s} {length:.2f} m")
    print("lane line to edge line (expected about 3.75 m; not used by the fit):")
    for name, distances in checks.edge_distance_m.items():
        print(f"  {name:15s} " + "  ".join(f"{d:.2f}" for d in distances))
    print_zones(result)


if __name__ == "__main__":
    main()
