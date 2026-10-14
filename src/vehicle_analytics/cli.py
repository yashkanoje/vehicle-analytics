"""Command line entry point: run, calibrate, challans and dashboard."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from importlib.util import find_spec
from pathlib import Path

from vehicle_analytics.calibrate import CalibrationError, calibration_yaml
from vehicle_analytics.challan import (
    ChallanError,
    generate_for_run,
    load_fines,
    run_output_folder,
)
from vehicle_analytics.config import ConfigError, load_config
from vehicle_analytics.pipeline import (
    FpsError,
    OutputError,
    VideoFileSource,
    VideoOpenError,
    prepare_out_dir,
    run,
)
from vehicle_analytics.store import Store, StoreError
from vehicle_analytics.views import DB_ENV, ROOT_ENV

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
    run_parser.add_argument(
        "--db",
        type=Path,
        default=Path("data") / "analytics.db",
        help="SQLite file for vehicle records (default: data/analytics.db)",
    )
    run_parser.add_argument(
        "--fines", type=Path, help="YAML file of sections and penalties, printed on the challans"
    )

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

    challans_parser = commands.add_parser(
        "challans", help="draft or refresh the challan PDFs of a stored run"
    )
    challans_parser.add_argument("--run", type=int, required=True, help="run id in the database")
    challans_parser.add_argument("--db", type=Path, default=Path("data") / "analytics.db")
    challans_parser.add_argument("--out", type=Path, help="default: runs/<video name of the run>")
    challans_parser.add_argument("--fines", type=Path)

    dashboard_parser = commands.add_parser(
        "dashboard", help="browse the database and review challans in a local web app"
    )
    dashboard_parser.add_argument("--db", type=Path, default=Path("data") / "analytics.db")
    dashboard_parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="folder the runs were started from; stored file paths are relative to it "
        "(default: the current folder)",
    )
    return parser


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    fines = load_fines(args.fines) if args.fines else {}
    source = VideoFileSource(args.video)
    out_dir = prepare_out_dir(args.out or Path("runs") / args.video.stem)

    with Store(args.db) as store:
        # Imported here so that config, video and database errors surface before the models load.
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
            store=store,
        )
        # The run's results are reported before the challans are drafted, so that a
        # failure there does not hide them.
        print(f"frames: {result.frames_processed}  vehicles: {result.summary['total']}")
        print(f"outputs: {result.out_dir}")
        print(f"database: {args.db} (run {result.summary['run_id']})")
        challans = generate_for_run(store, result.summary["run_id"], fines, out_dir)
    print(f"violations: {len(result.violations)}  challans drafted: {len(challans)}")
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    print(calibration_yaml(args.video, args.width_m, args.length_m, args.points), end="")
    return 0


def _challans(args: argparse.Namespace) -> int:
    fines = load_fines(args.fines) if args.fines else {}
    with Store(args.db) as store:
        if not store.has_run(args.run):
            raise StoreError(f"no run {args.run} in {args.db}")
        video = next(row["video"] for row in store.runs() if row["id"] == args.run)
        # A redraft goes where the run's files already are; only a run that has none
        # falls back to the default folder for its video.
        earlier = run_output_folder(store.challan_details(args.run))
        out_dir = prepare_out_dir(args.out or earlier or Path("runs") / Path(video).stem)
        paths = generate_for_run(store, args.run, fines, out_dir)
    plural = "challan" if len(paths) == 1 else "challans"
    print(f"{len(paths)} {plural} in {out_dir / 'challans' / str(args.run)}")
    return 0


def _dashboard(args: argparse.Namespace) -> int:
    problem = None
    if find_spec("streamlit") is None:
        problem = 'the dashboard needs Streamlit: pip install "vehicle-analytics[dashboard]"'
    elif not args.db.is_file():
        problem = f"no database at {args.db}"  # opening it would create an empty one
    elif not args.root.is_dir():
        problem = f"--root {args.root} is not a folder"
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return USER_ERROR
    app = Path(__file__).with_name("dashboard.py")
    env = {**os.environ, DB_ENV: str(args.db.resolve()), ROOT_ENV: str(args.root.resolve())}
    command = [
        sys.executable, "-m", "streamlit", "run", str(app),
        # Local only: no listening on the network, no usage statistics, no first-run prompt.
        "--server.address", "localhost",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    print("Starting the dashboard: open the Local URL below. Ctrl-C stops it.")
    try:
        return subprocess.run(command, env=env, check=False).returncode
    except KeyboardInterrupt:  # Ctrl-C stops Streamlit; that is how the dashboard is closed
        return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="warning: %(message)s")
    args = build_parser().parse_args(argv)
    handlers = {
        "run": _run,
        "calibrate": _calibrate,
        "challans": _challans,
        "dashboard": _dashboard,
    }
    try:
        return handlers[args.command](args)
    except (
        ConfigError,
        VideoOpenError,
        FpsError,
        OutputError,
        StoreError,
        ChallanError,
        CalibrationError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return USER_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
