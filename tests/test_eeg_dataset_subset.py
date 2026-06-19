"""
Local unit tests for the EEG dataset's class-balanced subsampling
(`eb_jepa.datasets.eeg.dataset._balanced_subset`). Pure list logic — no EDF files,
no cluster needed. Validates the "5% of the data, 50/50 balanced" smoke-test setup.
"""
from eb_jepa.datasets.eeg.dataset import _balanced_subset


def _fake_items(n_normal, n_abnormal):
    items = [(f"normal_{i}.edf", 0) for i in range(n_normal)]
    items += [(f"abnormal_{i}.edf", 1) for i in range(n_abnormal)]
    return items


def test_no_subset_when_frac_is_none():
    items = _fake_items(80, 20)
    assert _balanced_subset(items, None, seed=0) == items


def test_subset_is_class_balanced():
    items = _fake_items(800, 200)  # imbalanced 80/20 source pool
    subset = _balanced_subset(items, frac=0.05, seed=0)

    labels = [label for _, label in subset]
    n0, n1 = labels.count(0), labels.count(1)
    assert n0 == n1 > 0
    # ~5% of 1000 total -> ~25 per class (round(0.05 * 1000 / 2))
    assert n0 == 25


def test_subset_is_deterministic_given_seed():
    items = _fake_items(100, 100)
    a = _balanced_subset(items, frac=0.1, seed=42)
    b = _balanced_subset(items, frac=0.1, seed=42)
    assert a == b


def test_subset_paths_are_a_subset_of_originals():
    items = _fake_items(50, 50)
    subset = _balanced_subset(items, frac=0.2, seed=1)
    assert set(subset) <= set(items)
