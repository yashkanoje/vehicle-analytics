"""Local dashboard over the analytics database, started by `vehicle-analytics dashboard`.

Streamlit runs this file as a script, top to bottom, on every interaction; do not import
it. It reads through store.py and changes one thing: a reviewer's decision on a challan,
with that challan's PDF reprinted to match.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from vehicle_analytics import views
from vehicle_analytics.challan import RULE_TITLES, STATUS_TEXT, ChallanError, review, when_text
from vehicle_analytics.store import Store, StoreError, resolve_path

PAGES = ("Overview", "Vehicles", "Violations")
FILTERS = (
    ("class", "Class"),
    ("colour", "Colour (unverified)"),
    ("lane", "Lane"),
    ("direction", "Direction"),
)


def show_image(stored: str | None, root: Path, caption: str) -> None:
    path = resolve_path(stored, root)
    if path is None:
        st.caption(f"{caption}: none saved")
    elif not path.is_file():
        st.warning(f"{caption} not found: {path}")
    else:
        st.image(str(path), caption=caption)


def bar_chart(counts: Mapping[Any, int], name: str, empty: str, **labels: str) -> None:
    if counts:
        st.bar_chart(pd.Series(counts, name=name), **labels)
    else:
        st.caption(empty)


def overview_page(store: Store, run: Mapping[str, Any]) -> None:
    summary = views.run_overview(run, store.vehicles(run["id"]), store.violations(run["id"]))
    length = f"{summary.duration_s:.1f} s" if summary.duration_s else "unknown"
    columns = st.columns(4)
    columns[0].metric("Vehicles counted", summary.counted)
    columns[1].metric("Video length", length)
    columns[2].metric("Violations", sum(summary.violations_by_rule.values()))
    columns[3].metric("Violators not counted", summary.uncounted)
    if summary.duration_s is None:
        st.info("This run did not record its length, so no flow rate is shown.")
    left, right = st.columns(2)
    with left:
        st.subheader("By direction")
        if summary.by_direction:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"direction": d, "vehicles": n, "per hour": summary.flow_per_hour.get(d)}
                        for d, n in summary.by_direction.items()
                    ]
                ),
                hide_index=True,
            )
        else:
            st.caption("No counted vehicles in this run.")
        st.subheader("By class")
        bar_chart(summary.by_class, "vehicles", "No counted vehicles in this run.")
    with right:
        st.subheader("Speed")
        bar_chart(
            views.speed_histogram(summary.speeds_kmph),
            "vehicles",
            "No measured speeds in this run.",
            x_label="km/h, lower edge of each 10 km/h band",
        )
        st.subheader("Violations by rule")
        bar_chart(summary.violations_by_rule, "violations", "No violations in this run.")


def vehicles_page(store: Store, run: Mapping[str, Any], root: Path) -> None:
    rows = store.vehicles(run["id"])
    if not rows:
        st.info("No vehicles stored for this run.")
        return
    columns = st.columns(len(FILTERS) + 1)
    chosen = {
        key: column.multiselect(
            label, views.options(rows, key), placeholder="All", key=f"filter_{key}_{run['id']}"
        )
        or None  # nothing picked: no filter
        for column, (key, label) in zip(columns, FILTERS, strict=False)
    }
    speeds = [row["speed_kmph"] for row in rows if row["speed_kmph"] is not None]
    speed_range = None
    if speeds and math.floor(min(speeds)) < math.ceil(max(speeds)):
        low, high = math.floor(min(speeds)), math.ceil(max(speeds))
        picked = columns[-1].slider(
            "Speed, km/h", low, high, (low, high), key=f"filter_speed_{run['id']}"
        )
        speed_range = None if picked == (low, high) else picked
    kept = views.filter_vehicles(
        rows,
        classes=chosen["class"],
        colours=chosen["colour"],
        lanes=chosen["lane"],
        directions=chosen["direction"],
        speed_range=speed_range,
    )
    hidden = "; vehicles with no measured speed are hidden while a range is set"
    note = hidden if speed_range else ""
    st.caption(f"{len(kept)} of {len(rows)} vehicles{note}")
    if not kept:
        st.info("No vehicle matches these filters.")
        return
    st.dataframe(pd.DataFrame(views.vehicle_table(kept)), hide_index=True)

    by_id = {row["id"]: row for row in kept}
    chosen_id = st.selectbox(
        "Show vehicle",
        list(by_id),
        format_func=lambda i: f"record {i}: track {by_id[i]['track_id']} ({by_id[i]['class']})",
        key=f"vehicle_{run['id']}",
    )
    vehicle = by_id[chosen_id]
    left, right = st.columns(2)
    with left:
        show_image(vehicle["snapshot_path"], root, "Snapshot")
    with right:
        for label, value in views.vehicle_table([vehicle])[0].items():
            st.text(f"{label}: {views.NOT_RECORDED if value is None else value}")
        found = [r for r in store.challan_details(run["id"]) if r["vehicle_id"] == chosen_id]
        st.subheader("Violations")
        if not found:
            st.caption("None.")
        for r in found:
            challan = r["challan_no"] or "no challan yet"
            st.text(f"{RULE_TITLES[r['rule']]}, {when_text(r)} ({challan})")


def violation_card(store: Store, run: Mapping[str, Any], row: Mapping[str, Any], root: Path):
    st.subheader(f"{row['challan_no'] or 'No challan yet'}: {RULE_TITLES[row['rule']]}")
    pictures, facts = st.columns([2, 1])
    with pictures:
        show_image(row["evidence_path"], root, "Frame at the violation")
        show_image(row["crop_path"], root, "Vehicle")
    with facts:
        st.text(f"Vehicle record {row['vehicle_id']}, track {row['track_id']} ({row['class']})")
        st.text(f"When: {when_text(row)}")
        if row["zone"]:
            st.text(f"Zone: {row['zone']}")
        if row["measured"] is not None:
            unit = row["unit"] or ""
            st.text(f"Measured: {row['measured']:.1f} {unit}")
            st.text(f"Limit: {row['limit_value']:.1f} {unit}")
            if row["margin"] is not None:
                st.text(f"Tolerance applied: {row['margin']:.1f} {unit}")
        if row["challan_no"] is None:
            return
        reviewed = f" on {row['reviewed_at'].replace('T', ' ')}" if row["reviewed_at"] else ""
        st.markdown(f"**Status:** {STATUS_TEXT[row['status']]}{reviewed}")
        pdf = resolve_path(row["pdf_path"], root)
        if pdf is not None and pdf.is_file():
            st.download_button(
                "Challan PDF",
                pdf.read_bytes(),
                file_name=pdf.name,
                mime="application/pdf",
                key=f"pdf_{row['violation_id']}",
            )
        else:
            st.warning(f"Challan PDF not found: {pdf}")
        approve, reject = st.columns(2)
        decision = None
        key = row["violation_id"]
        if approve.button("Approve", key=f"approve_{key}"):
            decision = "approved"
        if reject.button("Reject", key=f"reject_{key}"):
            decision = "rejected"
        if decision:
            try:
                review(store, run["id"], row["violation_id"], decision, root)
            except ChallanError as exc:
                st.session_state["review_warning"] = str(exc)
            st.rerun()


def violations_page(store: Store, run: Mapping[str, Any], root: Path, db: Path) -> None:
    warning = st.session_state.pop("review_warning", None)
    if warning:
        st.warning(warning)
    rows = store.challan_details(run["id"])
    if not rows:
        st.info("No violations in this run.")
        return
    undrafted = sum(row["challan_no"] is None for row in rows)
    if undrafted:
        st.info(
            f"{undrafted} of {len(rows)} violations have no challan yet. Draft them with "
            f"`vehicle-analytics challans --run {run['id']} --db {db}`, run from {root}."
        )
    for row in rows:
        with st.container(border=True):
            violation_card(store, run, row, root)


def main() -> None:
    st.set_page_config(page_title="Vehicle analytics", layout="wide")
    db = Path(os.environ.get(views.DB_ENV, str(Path("data") / "analytics.db")))
    root = Path(os.environ.get(views.ROOT_ENV, "."))
    st.sidebar.title("Vehicle analytics")
    if not db.is_file():  # Store() would create an empty database
        st.error(f"No database at {db}. Process a video first with `vehicle-analytics run`.")
        return
    try:
        with Store(db) as store:
            runs = store.runs()
            if not runs:
                st.info(f"No runs in {db} yet.")
                return
            by_id = {run["id"]: run for run in runs}
            run_id = st.sidebar.selectbox(
                "Run",
                list(by_id),
                index=len(runs) - 1,
                format_func=lambda i: f"{i}: {by_id[i]['video']} ({by_id[i]['started_at']})",
                key="run",
            )
            run = by_id[run_id]
            page = st.sidebar.radio("Page", PAGES, key="page")
            st.sidebar.caption(f"Database: {db}")
            st.sidebar.caption(f"Files from: {root}")
            st.title(page)
            if run["site_name"]:
                st.caption(run["site_name"])
            if page == "Overview":
                overview_page(store, run)
            elif page == "Vehicles":
                vehicles_page(store, run, root)
            else:
                violations_page(store, run, root, db)
    except StoreError as exc:
        st.error(str(exc))


main()
