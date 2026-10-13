from vehicle_analytics.detector import Detection
from vehicle_analytics.tracking import Track, to_deepsort_input, tracks_from_deepsort


class StubTrack:
    def __init__(self, track_id, ltrb, cls_name, confirmed=True, time_since_update=0):
        self.track_id = track_id
        self.time_since_update = time_since_update
        self._ltrb = ltrb
        self._cls_name = cls_name
        self._confirmed = confirmed

    def is_confirmed(self):
        return self._confirmed

    def to_ltrb(self):
        return self._ltrb

    def get_det_class(self):
        return self._cls_name


def test_detections_become_left_top_width_height():
    detections = [Detection((10.0, 20.0, 50.0, 80.0), 0.9, "car")]
    assert to_deepsort_input(detections) == [([10.0, 20.0, 40.0, 60.0], 0.9, "car")]


def test_confirmed_and_matched_tracks_are_kept():
    tracks = tracks_from_deepsort([StubTrack(7, [1, 2, 3, 4], "car")])
    assert tracks == [Track("7", (1.0, 2.0, 3.0, 4.0), "car")]


def test_tentative_tracks_are_dropped():
    assert tracks_from_deepsort([StubTrack(7, [1, 2, 3, 4], "car", confirmed=False)]) == []


def test_coasting_tracks_are_dropped():
    assert tracks_from_deepsort([StubTrack(7, [1, 2, 3, 4], "car", time_since_update=1)]) == []


def test_missing_class_falls_back_to_vehicle():
    assert tracks_from_deepsort([StubTrack(7, [1, 2, 3, 4], None)])[0].cls_name == "vehicle"
