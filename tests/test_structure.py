import numpy as np
import pandas as pd

from deepvision.features import structure as st


def test_merge_frames_order():
    even = np.arange(8, dtype=np.float32)[None, :, None] * 0 + np.array(st.FRAME_SETS["even"])[None, :, None]
    odd = np.array(st.FRAME_SETS["odd"], dtype=np.float32)[None, :, None]
    z = st.merge_frames(even, odd)
    assert z.shape == (1, 16, 1) and np.array_equal(z[0, :, 0], np.arange(16))


def test_structure_function_and_slope():
    # z(t) = t * e: raw S(k) = k^2 -> slope 2; a constant video has S = 0
    z = np.zeros((2, 16, 3), np.float32)
    z[0, :, 0] = np.arange(16)
    s = st.structure_function(z, normalise=False)
    assert s.shape == (2, 15)
    assert np.allclose(s[0], np.arange(1, 16) ** 2) and np.allclose(s[1], 0)
    assert np.isclose(st.loglog_slope(s[:1])[0], 2.0)
    # normalised: unit vectors on a circle, chord^2 = 2 - 2 cos(k * w)
    w = 0.1
    zc = np.stack([np.cos(w * np.arange(16)), np.sin(w * np.arange(16))], -1)[None].astype(np.float32) * 5
    sc = st.structure_function(zc, normalise=True)
    assert np.allclose(sc[0], 2 - 2 * np.cos(w * np.arange(1, 16)), atol=1e-5)


def test_video_kind():
    assert st.video_kind("Wan2.2-TI2V-5B-I", 1) == "I2V" and st.video_kind("HunyuanVideo-I2V", 1) == "I2V"
    assert st.video_kind("Wan2.2-TI2V-5B-T", 1) == "T2V" and st.video_kind("HunyuanVideo", 1) == "T2V"
    assert st.video_kind("sora-2", 1) == "closed" and st.video_kind("Panda-70M", 0) == "real"


def test_analyse_detects_faster_fakes():
    rng = np.random.default_rng(0)
    n = 60
    walk = lambda scale: np.cumsum(rng.normal(0, scale, (n, 16, 8)), axis=1) + 10  # noqa: E731
    z = np.concatenate([walk(1.0), walk(2.0)]).astype(np.float32)
    rows = pd.DataFrame({"label": [0] * n + [1] * n, "generator": ["R"] * n + ["G-T"] * n, "rate": 4,
                         "n_unique": 16, "kind": ["real"] * n + ["T2V"] * n})
    rows.loc[0, "n_unique"] = 15  # dropped
    res = st.analyse(z, rows, n_boot=200)
    d = res["diff"].query("metric == 'raw' and group == 'fake_all'")
    assert len(d) == 15 and d["ci_excludes_0"].all() and (d["fake_minus_real"] > 0).all()
    assert np.allclose(d["dt"], np.arange(1, 16) / 4)
    a = res["auroc"].query("metric == 'raw' and group == 'fake_all'").iloc[0]
    assert a["n_real"] == n - 1 and a["auroc_S1"] > 0.9
    assert set(res["curves"]["group"]) >= {"real", "fake_all", "kind=T2V", "generator=G-T"}
