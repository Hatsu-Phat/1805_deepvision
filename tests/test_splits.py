import pandas as pd
import pytest

from deepvision.data.splits import build_splits, check_splits, eval_set, split_counts


def index(prefix, counts, label_of):
    rows = []
    for gen, n in counts.items():
        for i in range(n):
            rows.append({"video_id": f"{prefix}{gen}_{i}", "key": f"{gen}/{prefix}{gen}_{i}", "shard": "s.tar",
                         "label": label_of(gen), "generator": gen})
    return pd.DataFrame(rows)


@pytest.fixture
def indices():
    train = index("tr_", {"youku": 500, "OpenSora": 1000, "SEINE": 300, "SVD": 200}, lambda g: int(g != "youku"))
    test = index("te_", {"MSR-VTT": 50, "Sora": 5}, lambda g: int(g != "MSR-VTT"))
    return train, test


def test_default_splits_train_on_everything_else(indices):
    train, test = indices
    s = build_splits(train, test, ood_real_frac=0.02, seed=2024)
    assert len(s) == len(train) + len(test) and s["video_id"].is_unique
    assert len(eval_set(s, "val_id")) == 0
    tr = eval_set(s, "train")
    assert tr.groupby("generator").size().to_dict() == {"OpenSora": 1000, "SVD": 200, "youku": 490}
    ood = eval_set(s, "val_ood")
    assert set(ood.query("label == 1")["generator"]) == {"SEINE"}
    assert len(ood.query("label == 0")) == 10 and not set(ood["video_id"]) & set(tr["video_id"])
    assert set(eval_set(s, "test")["video_id"]) == set(test["video_id"])
    counts = split_counts(s)
    assert counts["val_ood"] == {"videos": 310, "real": 10, "fake": 300, "by_generator": {"SEINE": 300, "youku": 10}}


def test_optional_val_id_is_stratified(indices):
    train, test = indices
    s = build_splits(train, test, val_frac=0.02)
    val = eval_set(s, "val_id")
    assert val.groupby("generator").size().to_dict() == {"OpenSora": 20, "SVD": 4, "youku": 10}  # 2% of 490 youku
    assert not set(val["video_id"]) & set(eval_set(s, "train")["video_id"])
    assert not set(val["video_id"]) & set(eval_set(s, "val_ood")["video_id"])


def test_splits_are_reproducible(indices):
    train, test = indices
    a = build_splits(train, test, seed=7)
    b = build_splits(train.sample(frac=1, random_state=0), test, seed=7)  # row order does not matter
    key = ["video_id", "split"]
    assert a[key].sort_values("video_id").reset_index(drop=True).equals(b[key].sort_values("video_id").reset_index(drop=True))
    assert not a[key].equals(build_splits(train, test, seed=8)[key])


def test_invalid_splits(indices):
    train, test = indices
    with pytest.raises(ValueError, match="not in train"):
        build_splits(train, test, holdout=("Nope",))
    with pytest.raises(ValueError, match="must be fake"):
        build_splits(train, test, holdout=("youku",))
    s = build_splits(train, test)
    with pytest.raises(ValueError, match="more than one split"):
        check_splits(pd.concat([s, s.iloc[:1].assign(split="test")]))


def test_no_holdout_trains_on_every_train_video(indices):
    train, test = indices
    s = build_splits(train, test, ood_real_frac=0.0, holdout=())
    assert len(eval_set(s, "val_ood")) == 0
    assert set(eval_set(s, "train")["video_id"]) == set(train["video_id"])
    assert split_counts(s)["train"]["videos"] == len(train)
