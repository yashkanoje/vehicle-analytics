"""Rule-based violation detection. Pure: a rule reads a TrackState and returns a Violation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from vehicle_analytics.geometry import Zone
from vehicle_analytics.state import TrackState

if TYPE_CHECKING:  # the config loader needs PyYAML; the rules do not
    from vehicle_analytics.config import RulesConfig


@dataclass(frozen=True)
class Violation:
    """One rule broken by one vehicle. The numbers are kept so the decision can be re-derived."""

    rule: str
    at_s: float
    measured: float | None = None
    limit_value: float | None = None
    margin: float | None = None
    unit: str | None = None
    zone: str | None = None


class Rule:
    """A rule fires at most once per vehicle."""

    name = ""

    def __init__(self) -> None:
        self._fired: set[str] = set()

    def update(self, state: TrackState, t: float) -> Violation | None:
        if state.track_id in self._fired:
            return None
        violation = self._check(state, t)
        if violation is not None:
            self._fired.add(state.track_id)
        return violation

    def forget(self, track_id: str) -> None:
        """Drop what is remembered about a vehicle that has left."""
        self._fired.discard(track_id)

    def confirm(self, state: TrackState, violation: Violation) -> Violation | None:
        """The violation as it stands once the vehicle has left, or None if it no longer does.

        A rule that depends on the vehicle's class fires on the class seen so far. The
        class kept in the record is the majority over the whole track, which can differ.
        """
        return violation

    def _check(self, state: TrackState, t: float) -> Violation | None:
        raise NotImplementedError


class Overspeed(Rule):
    """Zone speed above the class limit by more than the error margin.

    The margin, `sigma_margin` times the speed's own one-sigma error, is subtracted before
    the comparison, so measurement noise alone cannot produce a violation.
    """

    name = "overspeed"

    def __init__(self, limits_kmph: dict[str, float], sigma_margin: float) -> None:
        super().__init__()
        self.limits_kmph = limits_kmph
        self.sigma_margin = sigma_margin

    def _check(self, state: TrackState, t: float) -> Violation | None:
        if state.speed is None:
            return None
        limit = self.limits_kmph.get(state.cls_name, self.limits_kmph["default"])
        margin = self.sigma_margin * state.speed.err_kmph
        if state.speed.kmph - margin <= limit:
            return None
        return Violation(self.name, t, state.speed.kmph, limit, margin, "km/h")

    def confirm(self, state: TrackState, violation: Violation) -> Violation | None:
        limit = self.limits_kmph.get(state.cls_name, self.limits_kmph["default"])
        if violation.measured - violation.margin <= limit:
            return None
        return replace(violation, limit_value=limit)


class WrongWay(Rule):
    """Counted in a lane whose permitted direction is the other one."""

    name = "wrong_way"

    def __init__(self, zones: Sequence[Zone]) -> None:
        super().__init__()
        self._allowed = {z.name: z.allowed_direction for z in zones if z.kind == "lane"}

    def _check(self, state: TrackState, t: float) -> Violation | None:
        if state.direction is None or state.lane not in self._allowed:
            return None
        if state.direction == self._allowed[state.lane]:
            return None
        return Violation(self.name, t, zone=state.lane)


class _Dwell(Rule):
    """Shared by the rules that fire after a vehicle has stayed in a zone for a while."""

    def __init__(self, min_dwell_s: float, fps: float) -> None:
        super().__init__()
        self.min_dwell_s = min_dwell_s
        self.fps = fps

    def _zones_for(self, state: TrackState) -> Sequence[str]:
        raise NotImplementedError

    def _check(self, state: TrackState, t: float) -> Violation | None:
        for name in self._zones_for(state):
            entered = state.zone_entered.get(name)
            if entered is None:
                continue
            dwell = round(t - entered / self.fps, 6)
            if dwell >= self.min_dwell_s:
                return Violation(self.name, t, dwell, self.min_dwell_s, unit="s", zone=name)
        return None


class RestrictedZone(_Dwell):
    """Stayed on a hard shoulder or in a restricted area."""

    name = "restricted_zone"

    def __init__(self, zones: Sequence[Zone], min_dwell_s: float, fps: float) -> None:
        super().__init__(min_dwell_s, fps)
        self._names = [z.name for z in zones if z.kind in ("shoulder", "restricted")]

    def _zones_for(self, state: TrackState) -> Sequence[str]:
        return self._names


class ClassLane(_Dwell):
    """A vehicle class stayed in a lane that bans it."""

    name = "class_lane"

    def __init__(self, zones: Sequence[Zone], min_dwell_s: float, fps: float) -> None:
        super().__init__(min_dwell_s, fps)
        self._banned = {z.name: z.banned_classes for z in zones if z.banned_classes}

    def _zones_for(self, state: TrackState) -> Sequence[str]:
        return [name for name, banned in self._banned.items() if state.cls_name in banned]

    def confirm(self, state: TrackState, violation: Violation) -> Violation | None:
        banned = self._banned.get(violation.zone, ())
        return violation if state.cls_name in banned else None


class Stopped(Rule):
    """Stayed within a small radius in a lane for `min_stop_s`. Needs a calibration.

    Judged on how far the vehicle moved over the whole period, not on an instantaneous
    speed: far from the camera one pixel of box jitter is most of a metre, which over half
    a second reads as several km/h. The radius is the distance a vehicle at
    `stop_speed_kmph` would cover in `min_stop_s`.
    """

    name = "stopped"

    def __init__(
        self, zones: Sequence[Zone], stop_speed_kmph: float, min_stop_s: float, fps: float
    ) -> None:
        super().__init__()
        self._lanes = [z.name for z in zones if z.kind == "lane"]
        self.stop_speed_kmph = stop_speed_kmph
        self.min_stop_s = min_stop_s
        self.fps = fps

    def _check(self, state: TrackState, t: float) -> Violation | None:
        lane = next((name for name in self._lanes if name in state.zone_entered), None)
        history = state.ground_history
        if lane is None or not history or history[-1][0] != state.last_frame:
            # Not in a lane, or not fully visible now. Unknown counts as moving: a vehicle
            # is not accused on missing data.
            return None
        now_frame, here = history[-1]
        period_frames = self.min_stop_s * self.fps
        radius_m = self.stop_speed_kmph / 3.6 * self.min_stop_s
        sightings = 0
        for frame, ground in reversed(history):
            if math.dist(ground, here) > radius_m:
                return None
            sightings += 1
            if now_frame - frame >= period_frames:
                break
        else:
            return None  # not watched for long enough yet
        if sightings < period_frames / 2:
            return None  # seen too rarely in the period to say it stood still
        elapsed_s = (now_frame - frame) / self.fps
        measured = math.dist(ground, here) / elapsed_s * 3.6 if elapsed_s > 0 else 0.0
        return Violation(self.name, t, measured, self.stop_speed_kmph, unit="km/h", zone=lane)


def build_rules(
    config: RulesConfig,
    zones: Sequence[Zone],
    fps: float,
    calibrated: bool,
) -> tuple[list[Rule], dict[str, str]]:
    """The rules this scene can run, and why each of the others cannot."""
    lanes = [z for z in zones if z.kind == "lane"]
    off_road = [z for z in zones if z.kind in ("shoulder", "restricted")]
    rules: list[Rule] = []
    disabled: dict[str, str] = {}

    if config.overspeed is not None:
        rules.append(Overspeed(config.overspeed.limits_kmph, config.overspeed.sigma_margin))
    if config.wrong_way:
        if lanes:
            rules.append(WrongWay(zones))
        else:
            disabled["wrong_way"] = "the config has no lane zones"
    if config.restricted_zone_dwell_s is not None:
        if off_road:
            rules.append(RestrictedZone(zones, config.restricted_zone_dwell_s, fps))
        else:
            disabled["restricted_zone"] = "the config has no shoulder or restricted zones"
    if config.stopped is not None:
        if calibrated and lanes:
            stopped = config.stopped
            rules.append(Stopped(zones, stopped.stop_speed_kmph, stopped.min_stop_s, fps))
        else:
            disabled["stopped"] = "it needs a calibration and lane zones"
    if config.class_lane_dwell_s is not None:
        if any(z.banned_classes for z in lanes):
            rules.append(ClassLane(zones, config.class_lane_dwell_s, fps))
        else:
            disabled["class_lane"] = "no lane zone lists banned_classes"
    return rules, disabled
