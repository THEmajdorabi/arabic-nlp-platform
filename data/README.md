# Data

## What is used

**`amgadhasan/arabic_tweets_dialects`**, downloaded **2026-08-23**, **MIT
licence**. 147,725 Arabic tweets labelled with one of five country dialects.

| Class | Code | Rows | Share |
|---|---|---|---|
| Egyptian | EG | 57,636 | 39.02% |
| Libyan | LY | 36,499 | 24.71% |
| Lebanese | LB | 27,617 | 18.69% |
| Sudanese | SD | 14,434 | 9.77% |
| Moroccan | MA | 11,539 | 7.81% |

Imbalance ratio, largest over smallest: **4.99x**. This is why macro F1 and
per-class F1 are reported everywhere in this repo and accuracy is reported
nowhere: a model that answers "Egyptian" to everything is 39% accurate and has
a macro F1 of 0.11.

## What was considered and rejected

| Candidate | Why not |
|---|---|
| **QADI** (`arbml/QADI`) | Loads, but it is 540,590 rows of `tweet_id` and `label` with **no text**. Using it means rehydrating tweets through the API, with heavy attrition and a different corpus for every person who tries. Rejected for the same reason the build spec warns off NADI. |
| **`Abdelrahman-Rezk/Arabic_Dialect_Identification`** | 440k rows with real text and pre-made splits, and genuinely tempting. Rejected because its 18 classes are **integer ids with no label map** anywhere in the repo or card, and **no licence** is declared. A classifier whose class names are guesses is not one whose per-class F1 means anything. |
| **ArSarcasm v2** | `iabufarha/ArSarcasm-v2` is not resolvable on the Hub today. |
| **NADI**, **MADAR** | Registration or a licence agreement. Neither is a same-day `load_dataset()` call. |

## Splits

Made by this repo, not inherited: the upload ships a single `train` split.

- **Stratified**: so the 4.99x imbalance is preserved exactly in train,
  validation and test. Verified in `tests/test_splits_and_eval.py`.
- **Seeded** from `config.yaml` (`split.seed: 20260823`), so a split is
  reproducible from a committed file.
- 70 / 15 / 15, then the **training set only** is capped at 60,000 rows so a
  full three-seed run finishes in about an hour on one consumer GPU.
  Validation and test always cover the full 15%: capping before splitting would
  shrink the evaluation sets too and make every number a measurement of the cap.

| Split | Rows |
|---|---|
| train | 60,000 (capped from 103,407) |
| validation | 22,159 |
| test | 22,159 |

## Cleaning

`src/arabic_nlp/data/preprocess.py`, one function called by both training and
serving.

- `@handles` and URLs are **replaced with a token, not deleted**. They carry no
  dialect signal and a lot of vocabulary; left in, a TF-IDF model learns which
  accounts tweet which dialect, which is a leak that inflates validation and
  evaporates in production.
- The `#` is dropped and the hashtag word kept, because `#طال_عمره` is signal.
- Character runs are capped at 3: `ههههههههه` and `ههه` are the same token, and
  treating every run length as a distinct feature wastes a large slice of the
  vocabulary on nothing.
- Rows under 5 characters are dropped.
- Exact-duplicate texts are dropped **before** splitting, so a retweet cannot
  land in both train and test. This corpus had **0** duplicates, which is
  unusual for tweet data and suggests the uploader had already deduplicated.

Nothing was dropped by these filters on this corpus: 147,725 rows in, 147,725
usable. The filters stay because the next corpus will not be so tidy.

## The test set

Read **once**, by `scripts/final_eval.py`, which refuses to run without an
explicit `--i-know-this-touches-the-test-set` flag.

**Times read: 1**, on 2026-08-23, after every model decision had been made on
validation.
