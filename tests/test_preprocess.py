"""Preprocessing, including the cases that leak or lose signal.

The one function `clean` is called by both training and serving, so every case
here is also a check that the two cannot drift apart.
"""

from __future__ import annotations

import pytest

from arabic_nlp.data.preprocess import (
    MENTION_TOKEN,
    URL_TOKEN,
    clean,
    collapse_repeats,
    is_usable,
    normalize_ar,
)

# ---- the leak this prevents ---------------------------------------------


def test_mentions_are_replaced_not_deleted():
    """`@AlmFaisal` carries no dialect signal but a lot of vocabulary.

    Left in, a TF-IDF model learns which accounts tweet which dialect, which
    is a leak that inflates validation and evaporates in production.
    """
    out = clean("@AlmFaisal شن الاخبار")
    assert "AlmFaisal" not in out
    assert MENTION_TOKEN in out
    assert "شن" in out


def test_urls_are_replaced():
    out = clean("شوف هذا https://t.co/abc123 حلو")
    assert "t.co" not in out
    assert URL_TOKEN in out


def test_hashtag_word_is_kept_and_the_hash_is_dropped():
    """`#طال_عمره` is dialect signal; the `#` and the underscore are not."""
    out = clean("#طال_عمره بينا")
    assert "#" not in out
    assert "طال" in out and "عمره" in out


# ---- elongation ----------------------------------------------------------


def test_repeated_characters_are_capped():
    assert collapse_repeats("ههههههههههه", 3) == "ههه"
    assert collapse_repeats("cooool", 3) == "coool"  # 4 o's -> 3
    assert collapse_repeats("cool", 3) == "cool"  # 2 o's, under the cap, untouched


def test_collapse_is_a_no_op_when_disabled():
    assert collapse_repeats("ههههههههه", 0) == "ههههههههه"


# ---- Arabic normalization ------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("الأَجْرُ الأَسَاسِيُّ", "الاجر الاساسي"),
        ("إجازة إضافية", "اجازه اضافيه"),
        ("آخر الشهر", "اخر الشهر"),
        ("على مقتضى", "علي مقتضي"),
        ("الأجـــر", "الاجر"),
        ("المادة ٧٥", "الماده 75"),
        ("", ""),
    ],
)
def test_normalize_ar(raw, expected):
    assert normalize_ar(raw) == expected


def test_normalization_is_idempotent():
    """Applied twice must equal applied once.

    Training normalizes text that may already be normalized; serving
    normalizes fresh input. A non-idempotent normalizer makes those two paths
    disagree depending on how many times the text passed through.
    """
    for raw in ["الأَجْرُ الأَسَاسِيُّ", "إجازة", "المادة ٧٥", "mixed نص text"]:
        once = normalize_ar(raw)
        assert normalize_ar(once) == once


def test_normalization_is_off_by_default_in_clean():
    """MARBERT was pretrained on lightly processed text; see DECISIONS.md."""
    assert clean("الأجر") == "الأجر"
    assert clean("الأجر", normalize_arabic=True) == "الاجر"


# ---- filtering -----------------------------------------------------------


def test_short_texts_are_rejected():
    assert not is_usable("مرحبا", 10)
    assert is_usable("مرحبا كيف الحال", 10)


def test_clean_handles_empty_and_whitespace():
    assert clean("") == ""
    assert clean("    ") == ""


def test_clean_collapses_whitespace():
    assert clean("كلمة    ثانية\n\nثالثة") == "كلمة ثانية ثالثة"
