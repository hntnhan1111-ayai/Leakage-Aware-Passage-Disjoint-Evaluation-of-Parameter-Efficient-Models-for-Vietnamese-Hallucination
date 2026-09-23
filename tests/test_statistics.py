import numpy as np

from vihallu_repro.stats import mcnemar_exact, paired_bootstrap


def test_statistics_are_deterministic():
    y = np.array(["no", "intrinsic", "extrinsic", "no", "intrinsic", "extrinsic"])
    a = np.array(["no", "intrinsic", "extrinsic", "no", "intrinsic", "no"])
    b = np.array(["no", "extrinsic", "extrinsic", "no", "no", "extrinsic"])
    first = paired_bootstrap(y, {"a": a, "b": b}, iterations=100, seed=7)
    second = paired_bootstrap(y, {"a": a, "b": b}, iterations=100, seed=7)
    assert first == second
    result = mcnemar_exact(y, a, b)
    assert 0.0 <= result["exact_p_value"] <= 1.0
