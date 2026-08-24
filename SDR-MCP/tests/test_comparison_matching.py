import random
import tracemalloc

from rf_mcp.comparison import _classifications, _frequency_candidate_pairs


def _signals(frequencies):
    return [{"frequency_hz": frequency, "relative_power_db": -10.0} for frequency in frequencies]


def _reference_pairs(left, right, tolerance):
    pairs = [
        (abs(float(r["frequency_hz"]) - float(l["frequency_hz"])), li, ri)
        for li, l in enumerate(left)
        for ri, r in enumerate(right)
        if abs(float(r["frequency_hz"]) - float(l["frequency_hz"])) <= tolerance
    ]
    return sorted(pairs, key=lambda pair: (pair[0], pair[1], pair[2]))


def _reference_classifications(result, signals, tolerance):
    completed = [
        (index, item)
        for index, item in enumerate(result.get("classifications", []))
        if item.get("status") == "completed" and item.get("frequency_hz") is not None
    ]
    mapping = {}
    for signal_index, signal in enumerate(signals):
        frequency = float(signal["frequency_hz"])
        nearby = [
            (index, item)
            for index, item in completed
            if abs(float(item["frequency_hz"]) - frequency) <= tolerance
        ]
        if nearby:
            mapping[signal_index] = min(
                nearby,
                key=lambda entry: (
                    abs(float(entry[1]["frequency_hz"]) - frequency), entry[0]
                ),
            )[1]
    return mapping


def test_candidate_pairs_match_cross_product_on_randomized_small_inputs():
    randomizer = random.Random(20260824)
    for _ in range(250):
        left = _signals([randomizer.randrange(0, 10) * 50 for _ in range(randomizer.randrange(9))])
        right = _signals([randomizer.randrange(0, 10) * 50 for _ in range(randomizer.randrange(9))])
        tolerance = randomizer.choice([0, 49, 50, 51, 100])
        assert list(_frequency_candidate_pairs(left, right, tolerance)) == _reference_pairs(
            left, right, tolerance
        )


def test_candidate_pairs_cover_duplicates_ties_boundaries_unsorted_and_empty_sides():
    left = _signals([200, 100, 200])
    right = _signals([250, 150, 200, 200])
    assert list(_frequency_candidate_pairs(left, right, 50)) == _reference_pairs(left, right, 50)
    assert list(_frequency_candidate_pairs([], right, 50)) == []
    assert list(_frequency_candidate_pairs(left, [], 50)) == []


def test_classifications_match_reference_with_randomized_unsorted_ties():
    randomizer = random.Random(731)
    for _ in range(100):
        signals = _signals([randomizer.randrange(8) * 50 for _ in range(randomizer.randrange(8))])
        classifications = [
            {
                "frequency_hz": randomizer.randrange(8) * 50,
                "status": randomizer.choice(["completed", "failed"]),
                "best_label": str(index),
            }
            for index in range(randomizer.randrange(10))
        ]
        result = {"classifications": classifications}
        tolerance = randomizer.choice([0, 50, 100])
        assert _classifications(result, signals, tolerance) == _reference_classifications(
            result, signals, tolerance
        )


def test_maximum_signal_count_does_not_materialize_candidate_cross_product():
    # Scans support 500 signals per side. All duplicates are the worst candidate-count case.
    maximum = 500
    left = _signals([100] * maximum)
    right = _signals([100] * maximum)
    tracemalloc.start()
    candidates = _frequency_candidate_pairs(left, right, 50)
    first = next(candidates)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert first == (0.0, 0, 0)
    assert peak < 1_000_000
