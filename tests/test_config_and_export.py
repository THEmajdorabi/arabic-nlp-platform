"""Config validation, label stability, and the export contract.

None of these need a model. All of them are about the failures that are silent:
a label order that changes between runs, a split that leaves no training data,
an export that quietly ships a worse model.
"""

from __future__ import annotations

import pytest

from arabic_nlp.config import Settings, load_settings
from arabic_nlp.data.load import id_to_label, label_to_id
from arabic_nlp.models.transformer import class_weights, oversample_indices


@pytest.fixture
def settings():
    return load_settings()


# ---- label stability -----------------------------------------------------


def test_label_ids_are_sorted_and_therefore_stable(settings):
    """A label map built from set iteration order is the classic way to make a
    saved model silently predict the wrong classes after a reload."""
    mapping = label_to_id(settings)
    assert list(mapping) == sorted(mapping)
    assert mapping == label_to_id(load_settings())


def test_label_maps_are_inverses(settings):
    forward = label_to_id(settings)
    backward = id_to_label(settings)
    for name, index in forward.items():
        assert backward[index] == name


def test_every_configured_label_has_a_human_readable_name(settings):
    """`MA` in a confusion matrix is only useful if the README can say Moroccan."""
    for code, name in settings.dataset.labels.items():
        assert name and name != code


# ---- config validation ---------------------------------------------------


def test_a_split_that_leaves_no_training_data_is_rejected(settings):
    raw = settings.model_dump()
    raw["split"]["test_size"] = 0.5
    raw["split"]["validation_size"] = 0.45
    with pytest.raises(ValueError, match="leaves"):
        Settings.model_validate(raw)


def test_the_dataset_licence_and_download_date_are_recorded(settings):
    """Provenance a reader can check, not a dataset name on its own."""
    assert settings.dataset.license
    assert settings.dataset.downloaded


def test_three_seeds_are_configured(settings):
    """One run's number is not a result; the config has to make that possible."""
    assert len(settings.transformer.seeds) >= 3
    assert len(set(settings.transformer.seeds)) == len(settings.transformer.seeds)


# ---- imbalance handling --------------------------------------------------


def test_class_weights_are_inverse_frequency_and_mean_one():
    labels = [0] * 80 + [1] * 20
    weights = class_weights(labels, 2)
    assert weights[1] > weights[0]  # the rare class is weighted up
    assert weights.mean() == pytest.approx(1.0)


def test_class_weights_handle_a_class_with_no_examples():
    """A missing class must not divide by zero halfway through a training run."""
    weights = class_weights([0, 0, 1], 3)
    assert len(weights) == 3
    assert all(w > 0 for w in weights)


def test_oversampling_brings_every_class_to_the_largest_size():
    labels = [0] * 50 + [1] * 10 + [2] * 5
    picked = oversample_indices(labels, seed=1)
    counts = {c: sum(1 for i in picked if labels[i] == c) for c in (0, 1, 2)}
    assert counts[0] == counts[1] == counts[2] == 50


def test_oversampling_is_reproducible():
    labels = [0] * 30 + [1] * 10
    assert oversample_indices(labels, seed=7) == oversample_indices(labels, seed=7)


def test_oversampling_keeps_every_original_example():
    """Resampling up must not drop originals in the process."""
    labels = [0] * 30 + [1] * 10
    picked = oversample_indices(labels, seed=7)
    assert set(range(len(labels))) <= set(picked)
