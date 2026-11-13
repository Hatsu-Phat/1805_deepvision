"""Tests of features.scout (no fixtures, so scripts/run_tests.py can run them where pytest is missing)."""
import numpy as np
import pandas as pd
import torch

from deepvision.eval.metrics import vif_paired_protocol
from deepvision.features.scout import (METHODS, SMNet, PerturbedTopK, clip_differences, hard_topk,
                                       paired_bacc_boot, split_val)


def test_perturbed_topk_forward_and_gradient_sign():
    torch.manual_seed(0)
    s = torch.zeros(4, 16, requires_grad=True)
    p = PerturbedTopK.apply(s, 3, 200, 0.5)
    assert p.shape == (4, 3, 16)
    assert torch.allclose(p.sum(-1), torch.ones(4, 3))                 # every slot picks one frame
    assert torch.allclose(p.sum((1, 2)), torch.full((4,), 3.0))
    target = torch.zeros(16)
    target[[2, 7]] = 1
    (-(p.sum(1) * target).sum()).backward()                            # loss falls when frames 2 and 7 are picked
    g = s.grad.mean(0)
    assert g[[2, 7]].max() < 0 and g[[2, 7]].mean() < g[[0, 1, 3, 4]].mean()


def test_hard_topk_sorted():
    s = torch.tensor([[0.0, 5, 1, 4, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9]])
    idx = hard_topk(s, 3).argmax(-1)
    assert idx.tolist() == [[1, 3, 15]]


def test_clip_differences_scale_invariant():
    torch.manual_seed(0)
    z = torch.randn(2, 16, 8)
    a1, a2 = clip_differences(z)
    b1, b2 = clip_differences(3.0 * z + 1.0)
    assert torch.allclose(a1, b1, atol=1e-5) and torch.allclose(a2, b2, atol=1e-5)
    assert torch.all(a1[:, 0] == 0) and torch.all(a2[:, :2] == 0)


def test_every_method_runs():
    torch.manual_seed(0)
    z = torch.randn(5, 16, 12).half()
    for m in METHODS:
        net = SMNet(m, 12, torch.zeros(12), torch.ones(12), k=4, hidden=16, scout_hidden=8, n_samples=20)
        net.train()
        out = net(z)
        assert out.shape == (5,)
        out.sum().backward()
        net.eval()
        logit, w = net(z, return_weights=True)
        assert logit.shape == (5,)
        if m in ("u8_mean",):
            assert w.shape == (5, 8, 16)
        if m in ("topk_inst", "rand_attn", "scout_attn"):
            assert w.shape == (5, 4, 16) and torch.all(w.sum(1).max(-1).values <= 1)
        for sel in ("even", "random"):
            if m == "scout_attn":
                net.selection = sel
                assert net(z).shape == (5,)
                net.selection = None


def test_scout_gets_gradient():
    torch.manual_seed(0)
    net = SMNet("scout_attn", 12, torch.zeros(12), torch.ones(12), k=4, hidden=16, scout_hidden=8, n_samples=50)
    net.train()
    net(torch.randn(6, 16, 12)).sum().backward()
    assert net.scout.out.weight.grad is not None and net.scout.out.weight.grad.abs().sum() > 0


def test_split_val_keeps_names_together():
    rows = pd.DataFrame({"video_id": [f"v{i // 3}" for i in range(3000)], "label": [int(i % 3 > 0) for i in range(3000)]})
    m = split_val(rows, 0.1)
    assert rows.assign(m=m).groupby("video_id")["m"].nunique().max() == 1
    assert 0.05 < m.mean() < 0.15


def test_paired_boot_matches_protocol():
    rng = np.random.default_rng(0)
    names = [f"n{i}" for i in range(60)]
    reals = pd.DataFrame({"video_id": names, "label": 0, "generator": "real", "score": rng.uniform(0, 0.8, 60)})
    fakes = pd.concat([pd.DataFrame({"video_id": names, "label": 1, "generator": g, "score": rng.uniform(0.2, 1, 60)})
                       for g in ("a", "b")])
    df = pd.concat([reals, fakes], ignore_index=True)
    ref = vif_paired_protocol(df)["avg_bacc"]
    boot = paired_bacc_boot(df, 500)
    assert abs(boot.mean() - ref) < 0.02 and boot.std() > 0


def test_scout_finds_planted_frames():
    """Fakes carry a shift on 2 random frames out of 16; scout_attn with K = 2 must learn to pick them."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n, d = 512, 16
    z = torch.randn(n, 16, d) * 0.3
    y = torch.tensor(rng.integers(0, 2, n)).float()
    planted = torch.zeros(n, 16, dtype=torch.bool)
    for i in np.where(y.numpy() == 1)[0]:
        planted[i, rng.choice(16, 2, replace=False)] = True
    z[planted] += 2.0
    net = SMNet("scout_attn", d, z.mean((0, 1)), z.std((0, 1)), k=2, hidden=32, scout_hidden=16, n_samples=100)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    for step in range(300):
        net.train()
        net.sigma = max(0.02, 0.5 * (1 - step / 300))
        b = torch.as_tensor(rng.choice(n, 128, replace=False))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(z[b]), y[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        logit, w = net(z, return_weights=True)
    acc = ((logit > 0).float() == y).float().mean().item()
    fake = y == 1
    hit = (w.sum(1).bool() & planted)[fake].sum(1).float().mean().item()   # planted frames picked (of 2)
    assert acc > 0.95, acc
    assert hit > 1.5, hit
