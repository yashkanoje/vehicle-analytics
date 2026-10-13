"""Command line entry point: `vehicle-analytics run` and `vehicle-analytics calibrate`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from vehicle_analytics.calibrate import CalibrationError, calibration_yaml
from vehicle_analytics.config import ConfigError, load_config
from vehicle_analytics.pipeline import (
    FpsError,
    OutputError,
    VideoFileSource,
    VideoOpenError,
    prepare_out_dir,
    run,
)

USER_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vehicle-analytics",
        description="Count vehicles and measure their speed from fixed-camera video.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="process a video")
    run_parser.add_argument("video", type=Path)
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--out", type=Path, help="output folder (default: runs/<video name>)")
    run_parser.add_argument("--no-display", action="store_true", help="do not open a window")

    calibrate_parser = commands.add_parser(
        "calibrate", help="print a calibration block from four points on a known road rectangle"
    )
    calibrate_parser.add_argument("video", type=Path)
    calibrate_parser.add_argument("--width-m", type=float, required=True)
    calibrate_parser.add_argument("--length-m", type=float, required=True)
    calibrate_parser.add_argument(
        "--points",
        help="'x,y x,y x,y x,y' in pixels: near-left, near-right, far-right, far-left; "
        "omit to click them",
    )
    return parser


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    source = VideoFileSource(args.video)
    out_dir = prepare_out_dir(args.out or Path("runs") / args.video.stem)

    # Imported here so that config and video errors surface before the models load.
    from vehicle_analytics.detector import YoloDetector
    from vehicle_analytics.tracking import DeepSortTracker

    detector = YoloDetector(config.model.weights, config.model.conf, config.model.classes)
    tracker = DeepSortTracker(
        config.tracker.max_age, config.tracker.n_init, config.tracker.max_cosine_distance
    )
    result = run(
        source,
        config,
        detector,
        tracker,
        out_dir,
        video_name=args.video.name,
        display=not args.no_display,
    )
    print(f"frames: {result.frames_processed}  vehicles: {result.summary['total']}")
    print(f"outputs: {result.out_dir}")
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    print(calibration_yaml(args.video, args.width_m, args.length_m, args.points), end="")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        return _calibrate(args)
    except (ConfigError, VideoOpenError, FpsError, OutputError, CalibrationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return USER_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
