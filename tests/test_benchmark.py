from optirouteai.evaluation.benchmark import compute_gap_percent


def test_compute_gap_percent():
    gap = compute_gap_percent(
        solver_distance=110.0,
        reference_distance=100.0,
    )

    assert gap == 10.0


def test_compute_gap_percent_zero_reference():
    gap = compute_gap_percent(
        solver_distance=110.0,
        reference_distance=0.0,
    )

    assert gap == 0.0