"""Load and validate the YAML run configuration.

Image coordinates in the file are fractions of frame width and height, so a
config does not depend on the video's resolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from vehicle_analytics.geometry import Homography, Line, Point, Zone

DEFAULT_CLASSES = ["car", "motorcycle", "bus", "truck"]
ZONE_KINDS = ("lane", "shoulder", "restricted")
_REQUIRED = object()


class ConfigError(ValueError):
    """The configuration is missing a key or holds an invalid value."""


@dataclass(frozen=True)
class ModelConfig:
    weights: str
    conf: float
    classes: tuple[str, ...]


@dataclass(frozen=True)
class TrackerConfig:
    max_age: int
    n_init: int
    max_cosine_distance: float


@dataclass(frozen=True)
class SpeedConfig:
    meters_between_lines: float | None
    distance_rel_uncertainty: float
    jitter_px: float
    live_speed_window_s: float


@dataclass(frozen=True)
class ZoneConfig:
    name: str
    kind: str
    polygon: tuple[Point, ...]
    allowed_direction: str | None
    banned_classes: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationConfig:
    image_points: tuple[Point, ...]
    ground_points_m: tuple[Point, ...]


@dataclass(frozen=True)
class Config:
    model: ModelConfig
    tracker: TrackerConfig
    lines: dict[str, tuple[Point, Point]]
    count_line: str
    direction_positive: str
    direction_negative: str
    speed: SpeedConfig
    calibration: CalibrationConfig | None
    fps_override: float | None
    zones: tuple[ZoneConfig, ...]
    site_name: str | None
    site_location: str | None
    recording_start: datetime | None
    raw: dict[str, Any]

    def lines_px(self, width: int, height: int) -> dict[str, Line]:
        return {
            name: Line(name, (p1[0] * width, p1[1] * height), (p2[0] * width, p2[1] * height))
            for name, (p1, p2) in self.lines.items()
        }

    def homography_px(self, width: int, height: int) -> Homography | None:
        if self.calibration is None:
            return None
        image_px = [(x * width, y * height) for x, y in self.calibration.image_points]
        return Homography.from_points(image_px, self.calibration.ground_points_m)

    def zones_px(self, width: int, height: int) -> list[Zone]:
        return [
            Zone(
                zone.name,
                zone.kind,
                tuple((x * width, y * height) for x, y in zone.polygon),
                zone.allowed_direction,
                zone.banned_classes,
            )
            for zone in self.zones
        ]

    def direction_label(self, side_to: int) -> str:
        return self.direction_positive if side_to > 0 else self.direction_negative


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _section(data: dict, key: str) -> dict:
    value = data.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"config key '{key}' must be a mapping")
    return value


def _number(section: dict, path: str, default: Any = _REQUIRED) -> Any:
    key = path.rsplit(".", 1)[-1]
    value = section.get(key)
    if value is None:
        if default is _REQUIRED:
            raise ConfigError(f"config key '{path}' is missing")
        return default
    if not _is_number(value):
        raise ConfigError(f"config key '{path}' must be a number, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ConfigError(f"config key '{path}' must be a finite number")
    return float(value)


def _require(condition: bool, path: str, rule: str) -> None:
    if not condition:
        raise ConfigError(f"config key '{path}' {rule}")


def _point(value: Any, path: str) -> Point:
    ok = isinstance(value, list | tuple) and len(value) == 2 and all(map(_is_number, value))
    _require(ok, path, "must be a pair of numbers")
    _require(all(map(math.isfinite, value)), path, "must be finite")
    return (float(value[0]), float(value[1]))


def _fraction_point(value: Any, path: str) -> Point:
    point = _point(value, path)
    inside = 0.0 <= point[0] <= 1.0 and 0.0 <= point[1] <= 1.0
    _require(inside, path, "must lie between 0 and 1 (fraction of the frame)")
    return point


def _parse_model(data: dict) -> ModelConfig:
    section = _section(data, "model")
    weights = section.get("weights", "yolo11s.pt")
    _require(isinstance(weights, str) and bool(weights), "model.weights", "must be a string")
    conf = _number(section, "model.conf", 0.4)
    _require(0.0 < conf <= 1.0, "model.conf", "must be greater than 0 and at most 1")
    classes = section.get("classes", DEFAULT_CLASSES)
    ok = isinstance(classes, list) and bool(classes) and all(isinstance(c, str) for c in classes)
    _require(ok, "model.classes", "must be a non-empty list of class names")
    return ModelConfig(weights=weights, conf=conf, classes=tuple(classes))


def _parse_tracker(data: dict) -> TrackerConfig:
    section = _section(data, "tracker")
    max_age = _number(section, "tracker.max_age", 30)
    n_init = _number(section, "tracker.n_init", 3)
    distance = _number(section, "tracker.max_cosine_distance", 0.3)
    _require(max_age >= 1, "tracker.max_age", "must be at least 1")
    _require(n_init >= 1, "tracker.n_init", "must be at least 1")
    _require(distance > 0.0, "tracker.max_cosine_distance", "must be greater than 0")
    return TrackerConfig(max_age=int(max_age), n_init=int(n_init), max_cosine_distance=distance)


def _parse_lines(data: dict) -> dict[str, tuple[Point, Point]]:
    raw = data.get("lines")
    if not isinstance(raw, dict):
        raise ConfigError("config key 'lines' is missing or is not a mapping")
    lines: dict[str, tuple[Point, Point]] = {}
    for name in ("a", "b"):
        if name not in raw:
            raise ConfigError(f"config key 'lines.{name}' is missing")
        segment = raw[name]
        ok = isinstance(segment, list | tuple) and len(segment) == 2
        _require(ok, f"lines.{name}", "must be two points")
        p1 = _fraction_point(segment[0], f"lines.{name}[0]")
        p2 = _fraction_point(segment[1], f"lines.{name}[1]")
        _require(p1 != p2, f"lines.{name}", "has zero length")
        lines[name] = (p1, p2)
    return lines


def _parse_directions(data: dict) -> tuple[str, str]:
    section = _section(data, "directions")
    positive = section.get("positive", "down")
    negative = section.get("negative", "up")
    ok = isinstance(positive, str) and isinstance(negative, str) and positive != negative
    _require(ok, "directions", "must give two different labels, 'positive' and 'negative'")
    return positive, negative


def _parse_speed(data: dict) -> SpeedConfig:
    section = _section(data, "speed")
    meters = _number(section, "speed.meters_between_lines", None)
    if meters is not None:
        _require(meters > 0.0, "speed.meters_between_lines", "must be greater than 0")
    relative = _number(section, "speed.distance_rel_uncertainty", 0.05)
    jitter = _number(section, "speed.jitter_px", 2.0)
    window = _number(section, "speed.live_speed_window_s", 0.5)
    _require(relative >= 0.0, "speed.distance_rel_uncertainty", "must not be negative")
    _require(jitter >= 0.0, "speed.jitter_px", "must not be negative")
    _require(window > 0.0, "speed.live_speed_window_s", "must be greater than 0")
    return SpeedConfig(
        meters_between_lines=meters,
        distance_rel_uncertainty=relative,
        jitter_px=jitter,
        live_speed_window_s=window,
    )


def _parse_calibration(data: dict) -> CalibrationConfig | None:
    if data.get("calibration") is None:
        return None
    section = _section(data, "calibration")
    points: dict[str, tuple[Point, ...]] = {}
    for key, parse in (("image_points", _fraction_point), ("ground_points_m", _point)):
        raw = section.get(key)
        ok = isinstance(raw, list | tuple) and len(raw) == 4
        _require(ok, f"calibration.{key}", "must hold exactly four points")
        points[key] = tuple(parse(p, f"calibration.{key}[{i}]") for i, p in enumerate(raw))
    try:
        Homography.from_points(points["image_points"], points["ground_points_m"])
    except ValueError as exc:
        raise ConfigError(f"config key 'calibration': {exc}") from exc
    return CalibrationConfig(
        image_points=points["image_points"], ground_points_m=points["ground_points_m"]
    )


def _parse_zones(data: dict, directions: tuple[str, str]) -> tuple[ZoneConfig, ...]:
    raw = data.get("zones")
    if raw is None:
        return ()
    _require(isinstance(raw, list), "zones", "must be a list")
    zones: list[ZoneConfig] = []
    names: set[str] = set()
    for i, item in enumerate(raw):
        where = f"zones[{i}]"
        _require(isinstance(item, dict), where, "must be a mapping")
        name = item.get("name")
        named = isinstance(name, str) and bool(name)
        _require(named, f"{where}.name", "must be a non-empty string")
        _require(name not in names, f"{where}.name", f"repeats the zone name '{name}'")
        names.add(name)
        kind = item.get("kind")
        _require(kind in ZONE_KINDS, f"{where}.kind", "must be 'lane', 'shoulder' or 'restricted'")
        polygon = item.get("polygon")
        ok = isinstance(polygon, list | tuple) and len(polygon) >= 3
        _require(ok, f"{where}.polygon", "must hold at least three points")
        points = tuple(
            _fraction_point(p, f"{where}.polygon[{j}]") for j, p in enumerate(polygon)
        )
        allowed = item.get("allowed_direction")
        banned = item.get("banned_classes", [])
        ok = isinstance(banned, list) and all(isinstance(c, str) for c in banned)
        _require(ok, f"{where}.banned_classes", "must be a list of class names")
        if kind == "lane":
            rule = f"must be '{directions[0]}' or '{directions[1]}' for a lane"
            _require(allowed in directions, f"{where}.allowed_direction", rule)
        else:
            _require(allowed is None, f"{where}.allowed_direction", "is only for lanes")
            _require(not banned, f"{where}.banned_classes", "is only for lanes")
        zones.append(ZoneConfig(name, kind, points, allowed, tuple(banned)))
    return tuple(zones)


def _optional_text(section: dict, path: str) -> str | None:
    value = section.get(path.rsplit(".", 1)[-1])
    _require(value is None or isinstance(value, str), path, "must be text")
    return value


def _parse_recording_start(data: dict) -> datetime | None:
    value = data.get("recording_start")
    if value is None:
        return None
    rule = "must be an ISO date and time, like 2025-12-08T17:30:00"
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ConfigError(f"config key 'recording_start' {rule}") from exc
    _require(isinstance(value, datetime), "recording_start", rule)
    return value


def parse_config(data: Any) -> Config:
    if not isinstance(data, dict):
        raise ConfigError("config file must contain a mapping at the top level")
    lines = _parse_lines(data)
    count_line = data.get("count_line", "b")
    known_line = isinstance(count_line, str) and count_line in lines
    _require(known_line, "count_line", "must be 'a' or 'b'")
    positive, negative = _parse_directions(data)
    speed = _parse_speed(data)
    calibration = _parse_calibration(data)
    if speed.meters_between_lines is None and calibration is None:
        raise ConfigError("config needs 'speed.meters_between_lines' or a 'calibration' block")
    fps_override = _number(data, "fps_override", None)
    if fps_override is not None:
        _require(fps_override > 0.0, "fps_override", "must be greater than 0")
    site = _section(data, "site")
    return Config(
        model=_parse_model(data),
        tracker=_parse_tracker(data),
        lines=lines,
        count_line=count_line,
        direction_positive=positive,
        direction_negative=negative,
        speed=speed,
        calibration=calibration,
        fps_override=fps_override,
        zones=_parse_zones(data, (positive, negative)),
        site_name=_optional_text(site, "site.name"),
        site_location=_optional_text(site, "site.location"),
        recording_start=_parse_recording_start(data),
        raw=data,
    )


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise ConfigError(f"config file must be saved as UTF-8: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file is not valid YAML: {path}: {exc}") from exc
    return parse_config(data)
