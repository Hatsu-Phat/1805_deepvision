import numpy as np
import torch

from deepvision.baselines.d3 import d3_scores


def test_d3_matches_reference_formula():
    g = torch.Generator().manual_seed(0)
    f = torch.randn(3, 16, 32, generator=g)
    s = d3_scores(f)
    x = f.numpy()
    d1 = np.linalg.norm(x[:, 1:] - x[:, :-1], axis=-1)
    d2 = d1[:, 1:] - d1[:, :-1]
    assert np.allclose(s["std_l2"].numpy(), d2.std(1, ddof=1), atol=1e-5)
    assert np.allclose(s["avg_l2"].numpy(), d2.mean(1), atol=1e-5)
    assert s["std_cos"].shape == (3,)


def test_d3_constant_speed_has_zero_score():
    t = torch.arange(16, dtype=torch.float32)[None, :, None]
    f = t * torch.ones(1, 1, 8)  # straight line at constant speed: d1 constant, d2 = 0
    assert float(d3_scores(f)["std_l2"][0]) < 1e-5
