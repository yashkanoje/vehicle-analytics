"""Draft challans: what one says, how it is laid out, and making them for a run.

A challan here is a draft for human review. It names a vehicle record and its photographs,
not an owner: no registration number is read.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import yaml
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from vehicle_analytics.config import RULE_NAMES, ConfigError
from vehicle_analytics.store import Store

logger = logging.getLogger(__name__)

WATERMARK = "DEMONSTRATION - NOT A LEGAL DOCUMENT"
NUMBER = re.compile(r"VA-\d+-(\d+)")
SPEED_MODES = {
    "homography": "Ground-plane calibration",
    "two_line": "Assumed distance between two lines (not calibrated)",
}
NO_SCHEDULE = "As per applicable schedule"
NOT_RECORDED = "Not recorded"
RULE_TITLES = {
    "overspeed": "Exceeding the speed limit",
    "wrong_way": "Driving against the permitted direction",
    "restricted_zone": "Driving or waiting on a hard shoulder or restricted area",
    "stopped": "Stopping on the carriageway",
    "class_lane": "Vehicle class not permitted in this lane",
}


class ChallanError(RuntimeError):
    """A challan PDF or its folder cannot be written."""


@dataclass(frozen=True)
class Fine:
    section: str | None
    description: str | None
    penalty: str | dict[str, str] | None
    source: str | None

    def penalty_for(self, cls_name: str) -> str | None:
        if isinstance(self.penalty, dict):
            return self.penalty.get(cls_name, self.penalty["default"])
        return self.penalty


def _text(entry: dict, rule: str, key: str) -> str | None:
    value = entry.get(key)
    if value is not None and not isinstance(value, str):
        raise ConfigError(f"fines file: '{rule}.{key}' must be text")
    return value


def load_fines(path: str | Path) -> dict[str, Fine]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"fines file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"fines file is not valid UTF-8 YAML: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("fines file must be a mapping of rule names")
    fines: dict[str, Fine] = {}
    for rule, entry in data.items():
        if rule not in RULE_NAMES:
            raise ConfigError(f"fines file: unknown rule '{rule}'")
        if not isinstance(entry, dict):
            raise ConfigError(f"fines file: '{rule}' must be a mapping")
        penalty = entry.get("penalty")
        if isinstance(penalty, dict):
            if "default" not in penalty:
                raise ConfigError(f"fines file: '{rule}.penalty' needs a 'default'")
            penalty = {str(cls_name): str(text) for cls_name, text in penalty.items()}
        elif penalty is not None and not isinstance(penalty, str):
            raise ConfigError(f"fines file: '{rule}.penalty' must be text or a mapping")
        fines[rule] = Fine(
            section=_text(entry, rule, "section"),
            description=_text(entry, rule, "description"),
            penalty=penalty,
            source=_text(entry, rule, "source"),
        )
    return fines


def _when(row: Mapping[str, Any]) -> str:
    if row["recording_start"]:
        moment = datetime.fromisoformat(row["recording_start"]) + timedelta(seconds=row["at_s"])
        return moment.strftime("%Y-%m-%d %H:%M:%S")
    return f"{row['at_s']:.1f} s from the start of the video"


def challan_fields(
    row: Mapping[str, Any], fine: Fine | None, challan_no: str, created_at: str
) -> list[tuple[str, str]]:
    """Label and value of every line on the challan, in order. `row` is from challan_details."""
    site = ", ".join(part for part in (row["site_name"], row["site_location"]) if part)
    fields = [
        ("Challan no.", challan_no),
        ("Status", "Pending review"),
        ("Drafted", created_at.replace("T", " ")),
        ("Violation", RULE_TITLES[row["rule"]]),
        ("Date and time", _when(row)),
        ("Site", site or NOT_RECORDED),
    ]
    if row["zone"]:
        fields.append(("Zone", row["zone"]))
    if row["measured"] is not None:
        unit = row["unit"] or ""
        fields.append(("Measured", f"{row['measured']:.1f} {unit}".strip()))
        fields.append(("Limit", f"{row['limit_value']:.1f} {unit}".strip()))
        if row["margin"] is not None:
            fields.append(("Tolerance applied", f"{row['margin']:.1f} {unit}".strip()))
    if row["rule"] == "overspeed" and row["speed_mode"] in SPEED_MODES:
        fields.append(("Speed measured by", SPEED_MODES[row["speed_mode"]]))
    fields += [
        ("Vehicle record", f"{row['vehicle_id']} (track {row['track_id']} in {row['video']})"),
        ("Vehicle class", row["class"]),
        ("Lane", row["lane"] or NOT_RECORDED),
        ("Direction", row["direction"] or NOT_RECORDED),
        ("Apparent colour (unverified)", row["colour"] or NOT_RECORDED),
        ("Registration number", "Not read"),
        ("Section", (fine and fine.section) or NO_SCHEDULE),
        ("Offence", (fine and fine.description) or RULE_TITLES[row["rule"]]),
        ("Penalty", (fine and fine.penalty_for(row["class"])) or NO_SCHEDULE),
    ]
    return fields


def _watermark(canvas: Any, _doc: Any) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 30)
    canvas.setFillColor(colors.Color(0.85, 0.2, 0.2, alpha=0.25))
    canvas.translate(A4[0] / 2, A4[1] / 2)
    canvas.rotate(45)
    canvas.drawCentredString(0, 0, WATERMARK)
    canvas.restoreState()


def _picture(path: str | None, width_mm: float) -> Image | None:
    if not path or not Path(path).is_file():
        return None
    picture = Image(path)
    scale = width_mm * mm / picture.imageWidth
    picture.drawWidth, picture.drawHeight = width_mm * mm, picture.imageHeight * scale
    return picture


def render_pdf(
    path: Path,
    fields: Sequence[tuple[str, str]],
    evidence_path: str | None,
    crop_path: str | None,
) -> None:
    styles = getSampleStyleSheet()
    table = Table(
        # Escaped: Paragraph reads its text as markup, and site, lane and file names are data.
        [
            [
                Paragraph(escape(label), styles["BodyText"]),
                Paragraph(escape(value), styles["BodyText"]),
            ]
            for label, value in fields
        ],
        colWidths=[60 * mm, 110 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story: list[Any] = [
        Paragraph("Traffic violation notice (draft for review)", styles["Title"]),
        Paragraph(
            "Produced by vehicle-analytics from video. It identifies a vehicle record and its "
            "photographs, not an owner. No registration number was read.",
            styles["Italic"],
        ),
        Spacer(1, 6 * mm),
        table,
        Spacer(1, 6 * mm),
    ]
    for caption, picture in (
        ("Evidence: the frame at the moment of the violation", _picture(evidence_path, 170)),
        ("Evidence: the vehicle", _picture(crop_path, 60)),
    ):
        if picture is not None:
            # A caption must not be left at the foot of a page with its picture on the next.
            story.append(KeepTogether([Paragraph(caption, styles["Heading4"]), picture]))
            story.append(Spacer(1, 4 * mm))
    document = SimpleDocTemplate(str(path), pagesize=A4, title=dict(fields)["Challan no."])
    document.build(story, onFirstPage=_watermark, onLaterPages=_watermark)


def run_output_folder(rows: Sequence[Mapping[str, Any]]) -> Path | None:
    """The output folder a run's challans or evidence were written to, from its stored paths."""
    for key in ("pdf_path", "evidence_path", "crop_path"):
        for row in rows:
            if row[key]:
                return Path(row[key]).parents[2]  # <out>/<challans|evidence>/<run_id>/<file>
    return None


def generate_for_run(
    store: Store, run_id: int, fines: Mapping[str, Fine], out_dir: Path
) -> list[Path]:
    """Draft a challan PDF for every violation of a run. Safe to run again: numbers are kept."""
    rows = store.challan_details(run_id)
    if not rows:
        return []
    folder = Path(out_dir) / "challans" / str(run_id)
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ChallanError(f"cannot write challans to {folder}: {exc}") from exc
    numbered = (NUMBER.fullmatch(row["challan_no"] or "") for row in rows)
    next_number = max((int(match.group(1)) for match in numbered if match), default=0) + 1
    now = datetime.now().isoformat(timespec="seconds")
    paths: list[Path] = []
    for row in rows:
        challan_no, created_at = row["challan_no"], row["created_at"]
        if challan_no is None:
            challan_no, created_at = f"VA-{run_id}-{next_number:04d}", now
            next_number += 1
        for image in (row["evidence_path"], row["crop_path"]):
            if image and not Path(image).is_file():
                logger.warning(
                    "challan %s: evidence image not found, left out: %s", challan_no, image
                )
        fine = fines.get(row["rule"])
        path = folder / f"{challan_no}.pdf"
        fields = challan_fields(row, fine, challan_no, created_at)
        try:
            render_pdf(path, fields, row["evidence_path"], row["crop_path"])
        except OSError as exc:
            raise ChallanError(f"cannot write challan {path}: {exc}") from exc
        store.save_challan(
            row["violation_id"],
            challan_no,
            created_at,
            fine.section if fine else None,
            fine.description if fine else None,
            fine.penalty_for(row["class"]) if fine else None,
            path.as_posix(),
        )
        paths.append(path)
    return paths
