"""Photo evidence for a violation: the frame with the vehicle outlined, and the vehicle crop."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

OUTLINE_COLOUR = (0, 0, 255)


def _write_jpeg(path: Path, image: np.ndarray) -> str | None:
    if image.size == 0:
        return None
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        return None
    # Written through Python, not cv2.imwrite, which fails on non-ASCII paths on Windows.
    path.write_bytes(encoded.tobytes())
    return path.as_posix()


def save_evidence(
    frame: np.ndarray, box_xyxy: Sequence[float], folder: Path, stem: str
) -> tuple[str | None, str | None]:
    """Write `<stem>_frame.jpg` and `<stem>_crop.jpg` in `folder`; the frame is not changed."""
    x1, y1, x2, y2 = (int(v) for v in box_xyxy)
    crop = frame[max(y1, 0) : y2, max(x1, 0) : x2]
    marked = frame.copy()
    cv2.rectangle(marked, (x1, y1), (x2, y2), OUTLINE_COLOUR, 3)
    return (
        _write_jpeg(folder / f"{stem}_frame.jpg", marked),
        _write_jpeg(folder / f"{stem}_crop.jpg", crop),
    )
