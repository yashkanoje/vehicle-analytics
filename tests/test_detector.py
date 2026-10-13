from vehicle_analytics.detector import Detection, detections_from_result

NAMES = {0: "person", 2: "car", 7: "truck"}


def test_only_allowed_classes_are_kept():
    detections = detections_from_result(
        xyxy=[[0, 0, 10, 10], [5, 5, 20, 30], [1, 2, 3, 4]],
        confs=[0.9, 0.8, 0.7],
        cls_ids=[0.0, 2.0, 7.0],
        names=NAMES,
        allowed={"car", "truck"},
    )
    assert detections == [
        Detection((5.0, 5.0, 20.0, 30.0), 0.8, "car"),
        Detection((1.0, 2.0, 3.0, 4.0), 0.7, "truck"),
    ]


def test_no_boxes_gives_no_detections():
    assert detections_from_result([], [], [], NAMES, {"car"}) == []
