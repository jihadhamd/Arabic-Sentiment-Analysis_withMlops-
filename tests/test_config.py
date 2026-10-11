from src.config import ID2LABEL, LABEL2ID, LABELS


def test_label_maps_are_consistent():
    assert len(LABELS) == 3
    for label in LABELS:
        assert ID2LABEL[LABEL2ID[label]] == label


def test_label_order_is_fixed():
    assert LABELS == ("negative", "mixed", "positive")
