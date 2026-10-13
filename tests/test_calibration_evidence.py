"""The calibration in configs/highway.yaml must follow from the committed measurements."""

import importlib.util
import sys
from pathlib import Path

import pytest

from vehicle_analytics.config import load_config

ROOT = Path(__file__).parent.parent
WIDTH, HEIGHT = 640, 360


def load_tool():
    path = ROOT / "tools" / "fit_highway_calibration.py"
    spec = importlib.util.spec_from_file_location("fit_highway_calibration", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return load_tool()


@pytest.fixture(scope="module")
def committed():
    return load_config(ROOT / "configs" / "highway.yaml")


def test_fit_reproduces_the_committed_calibration(tool, committed):
    result = tool.fit()
    assert result.rms_px == pytest.approx(1.01, abs=0.02)
    assert result.max_px == pytest.approx(1.89, abs=0.05)
    calibration = committed.calibration
    for got, want in zip(result.image_points, calibration.image_points, strict=True):
        assert got == pytest.approx(want, abs=2e-4)
    for got, want in zip(result.ground_points_m, calibration.ground_points_m, strict=True):
        assert got == pytest.approx(want, abs=0.011)


def test_documented_checks_hold_for_the_committed_calibration(tool, committed):
    checks = tool.check(committed.homography_px(WIDTH, HEIGHT))
    zone_lengths = list(checks.zone_length_m.values())
    assert len(zone_lengths) == 4
    assert min(zone_lengths) == pytest.approx(35.2, abs=0.06)
    assert max(zone_lengths) == pytest.approx(36.7, abs=0.06)
    edge_distances = [d for row in checks.edge_distance_m.values() for d in row]
    assert len(edge_distances) == 16
    assert min(edge_distances) == pytest.approx(3.62, abs=0.02)
    assert max(edge_distances) == pytest.approx(3.93, abs=0.02)
