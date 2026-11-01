import math

import numpy as np
import pytest

from deepvision.data.sampling import sampling_rate, uniform_indices, window_start


def demamba_rate(duration):
    # Preprocess/video2frame.py in chenhaoxing/DeMamba (8-frame rule)
    return 2 if duration >= 4 else math.ceil(8 / duration)


@pytest.mark.parametrize("duration", [0.5, 1.0, 2.0, 3.0, 3.99, 4.0, 6.5, 10.0, 60.0])
def test_eight_frame_rule_matches_demamba(duration):
    assert sampling_rate(duration, num_frames=8, base_fps=2) == demamba_rate(duration)


@pytest.mark.parametrize("duration,expected", [
    (1.0, 16), (2.0, 8), (3.0, 6), (4.0, 4), (5.0, 4), (7.9, 3), (8.0, 2), (10.0, 2), (120.0, 2),
])
def test_sixteen_frame_rule(duration, expected):
    assert sampling_rate(duration, num_frames=16, base_fps=2) == expected


@pytest.mark.parametrize("duration", [0.3, 1.0, 2.5, 4.0, 7.5, 8.0, 30.0])
def test_rate_yields_at_least_16_frames(duration):
    assert duration * sampling_rate(duration) >= 16 - 1e-9


def test_rate_rejects_non_positive_duration():
    with pytest.raises(ValueError):
        sampling_rate(0)


def test_window_start():
    assert window_start(10) == 0
    assert window_start(16) == 0
    assert window_start(20) == 2
    assert window_start(21) == 2
    assert window_start(240) == 112
    assert window_start(240, mode="first") == 0


def test_uniform_indices_8_of_16():
    assert uniform_indices(16, 8).tolist() == [0, 2, 4, 6, 9, 11, 13, 15]


def test_uniform_indices_short_clip():
    assert uniform_indices(5, 8).tolist() == [0, 1, 2, 3, 4]
    assert uniform_indices(0, 8).size == 0


def test_uniform_indices_sorted_and_unique():
    for n in range(8, 40):
        idx = uniform_indices(n, 8)
        assert len(idx) == 8 and np.all(np.diff(idx) > 0) and idx[0] == 0 and idx[-1] == n - 1
