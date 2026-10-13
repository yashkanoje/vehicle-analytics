
from vehicle_analytics.config import OverspeedConfig, RulesConfig, StoppedConfig
from vehicle_analytics.geometry import Zone
from vehicle_analytics.rules import (
    ClassLane,
    Overspeed,
    RestrictedZone,
    Stopped,
    Violation,
    WrongWay,
    build_rules,
)
from vehicle_analytics.speed import SpeedEstimate
from vehicle_analytics.state import TrackState

FPS = 10.0
SQUARE = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))
UP_LANE = Zone("up_1", "lane", SQUARE, "up")
FAST_LANE = Zone("down_3", "lane", SQUARE, "down", ("truck", "bus"))
SHOULDER = Zone("hs", "shoulder", SQUARE)
VERGE = Zone("verge", "restricted", SQUARE)


def track(cls_name="car", frame=100, zones=None, direction=None, lane=None, speed=None):
    state = TrackState("7")
    state.history.append((frame, (50.0, 50.0)))
    state.class_votes[cls_name] += 1
    state.zone_entered.update(zones or {})
    state.direction, state.lane, state.speed = direction, lane, speed
    return state


def speed(kmph, err):
    return SpeedEstimate(kmph=kmph, err_kmph=err, mode="homography")


def test_overspeed_reports_speed_limit_and_margin():
    rule = Overspeed({"default": 100.0}, sigma_margin=2.0)
    violation = rule.update(track(speed=speed(120.0, 5.0)), 12.5)
    assert violation == Violation(
        rule="overspeed", at_s=12.5, measured=120.0, limit_value=100.0, margin=10.0, unit="km/h"
    )


def test_overspeed_needs_the_whole_error_margin_above_the_limit():
    rule = Overspeed({"default": 100.0}, sigma_margin=2.0)
    assert rule.update(track(speed=speed(109.0, 5.0)), 1.0) is None  # 109 - 10 = 99
    assert rule.update(track(speed=speed(110.0, 5.0)), 1.0) is None  # exactly at the limit
    assert rule.update(track(speed=speed(110.1, 5.0)), 1.0) is not None


def test_overspeed_uses_the_limit_of_the_vehicle_class():
    rule = Overspeed({"default": 112.7, "truck": 96.6}, sigma_margin=2.0)
    assert rule.update(track("car", speed=speed(100.0, 1.0)), 1.0) is None
    violation = rule.update(track("truck", speed=speed(100.0, 1.0)), 1.0)
    assert violation.limit_value == 96.6


def test_overspeed_waits_for_a_measured_speed():
    assert Overspeed({"default": 100.0}, 2.0).update(track(), 1.0) is None


def test_a_rule_fires_once_per_vehicle_until_forgotten():
    rule = Overspeed({"default": 100.0}, sigma_margin=2.0)
    fast = track(speed=speed(150.0, 5.0))
    assert rule.update(fast, 1.0) is not None
    assert rule.update(fast, 1.1) is None
    rule.forget("7")
    assert rule.update(fast, 1.2) is not None


def test_wrong_way_compares_direction_with_the_lane():
    rule = WrongWay([UP_LANE, SHOULDER])
    assert rule.update(track(direction="up", lane="up_1"), 3.0) is None
    violation = rule.update(track(direction="down", lane="up_1"), 3.0)
    assert violation == Violation(rule="wrong_way", at_s=3.0, zone="up_1")
    assert rule.update(track(direction="down", lane=None), 3.0) is None
    assert rule.update(track(direction=None, lane="up_1"), 3.0) is None


def test_restricted_zone_needs_the_dwell_time():
    rule = RestrictedZone([SHOULDER, VERGE, UP_LANE], min_dwell_s=2.0, fps=FPS)
    inside = track(zones={"hs": 50})  # entered at frame 50 = 5.0 s
    assert rule.update(inside, 6.9) is None
    violation = rule.update(inside, 7.0)
    assert violation == Violation(
        rule="restricted_zone", at_s=7.0, measured=2.0, limit_value=2.0, unit="s", zone="hs"
    )


def test_restricted_zone_ignores_lanes():
    rule = RestrictedZone([SHOULDER, UP_LANE], min_dwell_s=2.0, fps=FPS)
    assert rule.update(track(zones={"up_1": 0}), 50.0) is None


def test_class_lane_flags_only_banned_classes_after_the_dwell_time():
    rule = ClassLane([FAST_LANE, UP_LANE], min_dwell_s=2.0, fps=FPS)
    assert rule.update(track("car", zones={"down_3": 0}), 9.0) is None
    assert rule.update(track("truck", zones={"up_1": 0}), 9.0) is None
    assert rule.update(track("truck", zones={"down_3": 80}), 9.9) is None
    violation = rule.update(track("truck", zones={"down_3": 80}), 10.0)
    assert violation == Violation(
        rule="class_lane", at_s=10.0, measured=2.0, limit_value=2.0, unit="s", zone="down_3"
    )


def metres(p):
    return (p[0] / 10.0, p[1] / 10.0)


def stopped_rule():
    return Stopped([UP_LANE], stop_speed_kmph=5.0, min_stop_s=3.0, fps=FPS)


def test_stopped_fires_after_standing_still_in_a_lane():
    rule, state = stopped_rule(), TrackState("7")
    fired = {}
    for frame in range(1, 41):
        state.observe(frame, (50.0, 50.0), "car", [], metres, [UP_LANE])
        violation = rule.update(state, frame / FPS)
        if violation is not None:
            fired[frame] = violation
    # Seen from frame 1, so three seconds of standing are complete at frame 31.
    assert list(fired) == [31]
    assert fired[31] == Violation(
        rule="stopped", at_s=3.1, measured=0.0, limit_value=5.0, unit="km/h", zone="up_1"
    )


def test_stopped_ignores_a_moving_vehicle_and_a_vehicle_outside_any_lane():
    rule, moving = stopped_rule(), TrackState("7")
    for frame in range(1, 41):  # 2 px = 0.2 m per frame = 7.2 km/h
        moving.observe(frame, (50.0, 10.0 + 2.0 * frame), "car", [], metres, [UP_LANE])
        assert rule.update(moving, frame / FPS) is None
    rule, parked = stopped_rule(), TrackState("8")
    for frame in range(1, 41):
        parked.observe(frame, (500.0, 500.0), "car", [], metres, [UP_LANE])
        assert rule.update(parked, frame / FPS) is None


def test_stopped_starts_again_after_the_vehicle_moves():
    rule, state = stopped_rule(), TrackState("7")
    fired = []
    for frame in range(1, 81):
        y = 50.0 if frame < 20 or frame >= 30 else 50.0 + 5.0 * (frame - 19)
        state.observe(frame, (50.0, y), "car", [], metres, [UP_LANE])
        if rule.update(state, frame / FPS) is not None:
            fired.append(frame)
    assert len(fired) == 1 and fired[0] > 50


FULL = RulesConfig(
    overspeed=OverspeedConfig({"default": 100.0}, 2.0),
    wrong_way=True,
    restricted_zone_dwell_s=2.0,
    stopped=StoppedConfig(5.0, 3.0),
    class_lane_dwell_s=2.0,
)


def test_no_rules_configured_means_no_rules():
    assert build_rules(RulesConfig(), [UP_LANE], FPS, True) == ([], {})


def test_all_rules_run_in_a_scene_that_supports_them():
    rules, disabled = build_rules(FULL, [UP_LANE, FAST_LANE, SHOULDER], FPS, True)
    names = [rule.name for rule in rules]
    assert names == ["overspeed", "wrong_way", "restricted_zone", "stopped", "class_lane"]
    assert disabled == {}


def test_rules_without_their_requirements_are_disabled():
    rules, disabled = build_rules(FULL, [], FPS, False)
    assert [rule.name for rule in rules] == ["overspeed"]
    assert set(disabled) == {"wrong_way", "restricted_zone", "stopped", "class_lane"}
    assert "lane" in disabled["wrong_way"]
    assert "calibration" in disabled["stopped"]


def test_class_lane_needs_a_lane_with_banned_classes():
    _, disabled = build_rules(FULL, [UP_LANE, SHOULDER], FPS, True)
    assert set(disabled) == {"class_lane"}
    assert "banned_classes" in disabled["class_lane"]


def far_away(p):
    """0.8 m per pixel, as at the far line of the motorway scene."""
    return (p[0] * 0.8, p[1] * 0.8)


def test_stopped_is_not_defeated_by_box_jitter_far_from_the_camera():
    # One pixel of jitter is 0.8 m here: over half a second that reads as 6 km/h.
    rule, state = stopped_rule(), TrackState("7")
    fired = []
    for frame in range(1, 61):
        y = 50.0 + (frame % 2)
        state.observe(frame, (50.0, y), "car", [], far_away, [UP_LANE])
        if rule.update(state, frame / FPS) is not None:
            fired.append(frame)
    assert fired == [31]


def test_stopped_does_not_judge_a_vehicle_that_is_no_longer_fully_visible():
    rule, state = stopped_rule(), TrackState("7")
    for frame in range(1, 7):
        state.observe(frame, (50.0, 50.0), "car", [], metres, [UP_LANE])
        assert rule.update(state, frame / FPS) is None
    for frame in range(7, 61):  # cut by the frame edge, and in fact moving
        state.observe(frame, (50.0, 50.0), "car", [], metres, [UP_LANE], visible=False)
        assert rule.update(state, frame / FPS) is None


def test_stopped_needs_enough_sightings_in_the_period():
    rule, state = stopped_rule(), TrackState("7")
    state.observe(1, (50.0, 50.0), "car", [], metres, [UP_LANE])
    state.observe(40, (50.0, 50.0), "car", [], metres, [UP_LANE])  # unseen for 3.9 s
    assert rule.update(state, 4.0) is None


def test_overspeed_is_dropped_when_the_final_class_has_a_higher_limit():
    rule = Overspeed({"default": 1000.0, "truck": 10.0}, sigma_margin=2.0)
    state = track("truck", speed=speed(100.0, 1.0))
    violation = rule.update(state, 1.0)
    assert violation.limit_value == 10.0
    assert rule.confirm(state, violation) == violation
    state.class_votes["car"] += 5  # the detector settles on "car"
    assert rule.confirm(state, violation) is None


def test_overspeed_keeps_the_violation_with_the_final_class_limit():
    rule = Overspeed({"default": 50.0, "truck": 10.0}, sigma_margin=2.0)
    state = track("truck", speed=speed(100.0, 1.0))
    violation = rule.update(state, 1.0)
    state.class_votes["car"] += 5
    assert rule.confirm(state, violation).limit_value == 50.0


def test_class_lane_is_dropped_when_the_final_class_is_allowed():
    rule = ClassLane([FAST_LANE], min_dwell_s=2.0, fps=FPS)
    state = track("truck", zones={"down_3": 0})
    violation = rule.update(state, 5.0)
    assert rule.confirm(state, violation) == violation
    state.class_votes["car"] += 5
    assert rule.confirm(state, violation) is None


def test_other_rules_are_confirmed_as_they_stand():
    rule = WrongWay([UP_LANE])
    state = track(direction="down", lane="up_1")
    violation = rule.update(state, 3.0)
    assert rule.confirm(state, violation) == violation


def test_rules_module_does_not_need_the_config_loader():
    import subprocess
    import sys

    code = "import sys, vehicle_analytics.rules; print('yaml' in sys.modules)"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert done.stdout.strip() == "False"
