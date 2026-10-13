import cv2
import numpy as np

from vehicle_analytics.evidence import save_evidence


def test_frame_and_crop_are_written(tmp_path):
    frame = np.full((100, 200, 3), 40, dtype=np.uint8)
    frame[30:60, 50:90] = (10, 200, 10)
    frame_path, crop_path = save_evidence(frame, (50.0, 30.0, 90.0, 60.0), tmp_path, "7_overspeed")
    assert frame_path == (tmp_path / "7_overspeed_frame.jpg").as_posix()
    assert crop_path == (tmp_path / "7_overspeed_crop.jpg").as_posix()
    marked, crop = cv2.imread(frame_path), cv2.imread(crop_path)
    assert marked.shape == (100, 200, 3) and crop.shape == (30, 40, 3)
    assert crop[:, :, 1].mean() > 150  # the vehicle itself, with no outline drawn into it
    blue, green, red = (int(v) for v in marked[30, 70])
    assert red > 150 and red > green + 60 and red > blue + 60  # outlined in red


def test_the_frame_given_is_not_drawn_on(tmp_path):
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    save_evidence(frame, (50.0, 30.0, 90.0, 60.0), tmp_path, "7_stopped")
    assert not frame.any()


def test_a_box_with_no_pixels_gives_a_frame_but_no_crop(tmp_path):
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame_path, crop_path = save_evidence(frame, (50.0, 30.0, 50.4, 60.0), tmp_path, "7_wrong_way")
    assert frame_path is not None and crop_path is None
