"""Text cleaning, and the Arabic normalizer.

Two things are deliberate here.

**Handles and URLs are replaced, not deleted.** `@AlmFaisal` carries no dialect
signal but it does carry vocabulary, and a TF-IDF model will happily learn that
a particular account tweets Libyan. Replacing them with a single token removes
the leak while leaving the sentence shape intact.

**Arabic normalization is a setting, not a default.** Folding hamza, tashkeel
and alef variants helps a bag-of-character-n-grams model, which cannot
generalise across spellings on its own. It plausibly *hurts* MARBERT, which was
pretrained on lightly processed text and has already learned those variants --
and in dialect identification the spelling variation is part of the signal. So
both are configurable, both are measured, and neither is assumed. See
DECISIONS.md.
"""

from __future__ import annotations

import re
import unicodedata

MENTION = re.compile(r"@\w+")
URL = re.compile(r"https?://\S+|www\.\S+")
HASHTAG_SYMBOL = re.compile(r"#(\w+)")
WHITESPACE = re.compile(r"\s+")

MENTION_TOKEN = "@مستخدم"
URL_TOKEN = "رابط"

# Tashkeel (U+064B..U+065F) plus the superscript alef (U+0670).
TASHKEEL = re.compile(r"[ً-ٰٟ]")
TATWEEL = re.compile(r"ـ")
# Arabic-Indic and extended Arabic-Indic digits.
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
ALEF = re.compile(r"[آأإٱ]")
YAA = re.compile(r"ى")
TAA_MARBUTA = re.compile(r"ة")


def normalize_ar(text: str) -> str:
    """Fold Arabic orthography to one spelling.

    Deliberately the same transformation as repo 02's `normalize_ar`, and
    deliberately a separate implementation: this repo does not import repo 02,
    which imports repo 01, which runs repo 03. `tests/test_normalize.py` pins
    the behaviour so the two cannot drift silently.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = TASHKEEL.sub("", text)
    text = TATWEEL.sub("", text)
    text = text.translate(ARABIC_DIGITS)
    text = ALEF.sub("ا", text)
    text = YAA.sub("ي", text)
    text = TAA_MARBUTA.sub("ه", text)
    return WHITESPACE.sub(" ", text).strip()


def collapse_repeats(text: str, limit: int) -> str:
    """`ههههههههه` -> `ههه`.

    Elongation is expressive, not lexical. Capping it stops the vectoriser
    treating every run length as a different feature, which on tweet data is a
    large and entirely uninformative slice of the vocabulary.
    """
    if limit <= 0:
        return text
    return re.sub(r"(.)\1{" + str(limit) + r",}", r"\1" * limit, text)


def clean(
    text: str,
    *,
    replace_mentions: bool = True,
    replace_urls: bool = True,
    strip_repeated_chars: int = 3,
    normalize_arabic: bool = False,
) -> str:
    """The one function both training and serving call.

    Training-time and serving-time preprocessing drifting apart is the single
    most common way a deployed classifier quietly loses accuracy, so there is
    exactly one of these and the service imports it rather than reimplementing
    the "obvious" parts.
    """
    if not text:
        return ""
    if replace_urls:
        text = URL.sub(URL_TOKEN, text)
    if replace_mentions:
        text = MENTION.sub(MENTION_TOKEN, text)
    # The hash is dropped, the word is kept: `#طال_عمره` is dialect signal.
    text = HASHTAG_SYMBOL.sub(r"\1", text)
    text = text.replace("_", " ")
    text = collapse_repeats(text, strip_repeated_chars)
    if normalize_arabic:
        text = normalize_ar(text)
    return WHITESPACE.sub(" ", text).strip()


def is_usable(text: str, min_chars: int) -> bool:
    """Rows too short to carry dialect signal are dropped, not guessed at."""
    return len(text.strip()) >= min_chars
