# Decisions

Every real choice between alternatives, with the evidence that decided it.
Entries later shown wrong are marked, not deleted.

## 2026-08-23, Dataset: `amgadhasan/arabic_tweets_dialects`

**Chosen:** 147,725 Arabic tweets, five country dialects, MIT licence,
downloadable today with one `load_dataset` call.

**Rejected, with reasons:**

- **QADI** (`arbml/QADI`) loads 540,590 rows of `tweet_id` and `label` and
  **no text**. Using it means rehydrating tweets through the API, with heavy
  attrition and a different corpus for every person who runs it. That is the
  exact failure the build spec warns about for NADI, and it applies here.
- **`Abdelrahman-Rezk/Arabic_Dialect_Identification`** has 440k rows of real
  text and ready-made splits, and was genuinely tempting. Rejected because its
  18 classes are **integer ids with no label map** anywhere in the repo or the
  card, and it declares **no licence**. Per-class F1 for a class whose name is
  a guess is not a result.
- **ArSarcasm v2** is not resolvable on the Hub today.
- **NADI** and **MADAR** need registration or a licence agreement.

**The trade this makes:** five classes rather than eighteen, so the task is
easier than country-level identification over the whole Arab world. It is not
trivial. EG/LY and MA/LY are genuinely confusable, and the confusion matrix
shows it.

## 2026-08-23: The baseline is built first, and it is a real baseline

**Chosen:** TF-IDF over **character 3-5 grams** (`char_wb`) plus word unigrams,
into logistic regression with `class_weight="balanced"`.

**Rejected:** starting with the transformer; a word-only bag of words.

**Why character n-grams specifically.** Arabic dialect markers are frequently
sub-word: the Levantine future prefix `ح-`, the Egyptian progressive `ب-`, the
Maghrebi negation circumfix `ما...ش`. A word-unigram model sees `حيروح` and
`رايح` as unrelated tokens; a character model sees the shared marker. Word
unigrams are kept alongside because the lexical giveaways, `برشا`, `زول`,
`عايز`, `واش`, are whole words that n-grams fragment.

**What it produced:** macro F1 **0.8203** on validation, in **47 seconds on a
CPU**. The highest-weighted features per class came out as textbook dialect
markers: `عم شو هيك رح هيدا بدو` for Lebanese, `واش ديال باش حيت راه` for
Moroccan, `زول ياخ ديل منو` for Sudanese, `شن توا هكي` for Libyan, `ده دي
علشان كام` for Egyptian. That is the cheapest sanity check that exists on a
labelled dataset, and it passed, which is worth more than the F1, because it
says the labels are real.

**Why it matters that it is this good.** A fine-tuned transformer reported at
0.86 with no baseline says nothing. Reported against 0.82 from 47 seconds of
CPU, it says what the GPU hour bought.

## 2026-08-23. Accuracy is not reported anywhere in this repo

**Chosen:** macro F1 and **per-class** precision, recall and F1, everywhere.

**Why:** the corpus runs 39.0% Egyptian to 7.8% Moroccan, a 4.99x imbalance. A
model that answers "Egyptian" to everything is **39% accurate** and has a
**macro F1 of 0.11**. Only one of those two numbers tells you it is useless,
and there is a test, `test_macro_f1_exposes_a_majority_class_predictor`,
that constructs exactly that model and asserts both.

## 2026-08-23, [MEASURED, CONTRADICTS THE INITIAL GUESS] Arabic normalization hurts the baseline

**What was assumed:** folding hamza, tashkeel, alef and taa marbuta would help
a character-n-gram model, which cannot generalise across spellings on its own.
The config comment said so before it was measured.

**What was measured**, same split, same seed, same everything else:

| Baseline | macro F1 | 95% CI | weighted F1 | ECE |
|---|---|---|---|---|
| `normalize_arabic: true` | 0.8162 | [0.8103, 0.8222] | 0.8468 | 0.0386 |
| `normalize_arabic: false` | **0.8203** | [0.8146, 0.8265] | 0.8506 | 0.0388 |

**And then the part that matters.** Those two intervals overlap across almost
their whole width, and the first conclusion written here was "0.004 on one
split is not a finding". That conclusion was **wrong**, and it was wrong in the
specific way this whole project keeps warning about: two overlapping
independent intervals do **not** mean no difference.

The paired test, resampling the same validation examples for both models:

```
raw minus normalized: +0.0041   95% CI [+0.0009, +0.0074]
interval excludes zero: True
```

Pairing removes the example-to-example variance, which is most of the variance,
and the difference survives. It is small, four thousandths of macro F1, and
it is real. `artifacts/baseline/normalization_ablation.json` holds the numbers.

**The explanation that fits:** in dialect identification, spelling variation is
*part of the signal*. Egyptian and Levantine writers differ in how they render
hamza and final yaa, and folding those away removes a feature rather than
adding robustness. This is the opposite of the situation in repo 02, where the
task is matching a query to a legal text and the same variation is pure noise.

**What changed as a result:** normalization is **off** for the transformer path
(`preprocess.normalize_arabic: false`), which was already the plan for a
separate reason. MARBERT was pretrained on lightly processed text. The
baseline keeps its own setting so both arms remain runnable.

**What is still NOT claimed:** that this generalises. One corpus, one split,
one model family, and an effect of 0.004. It is enough to justify the default
and not enough to justify a rule.

## 2026-08-23. Imbalance: class weights in the loss, and the alternatives are runnable

**Chosen:** inverse-frequency class weights in the cross-entropy, normalised to
mean 1.

**Why normalised:** it keeps the loss on the same scale as the unweighted run,
so the learning rate does not have to change between strategies and the
comparison stays a comparison of the strategy rather than of two learning
rates.

**Why the alternatives stay in the code:** `oversample` and `none` are both
selectable from `config.yaml` and from the CLI. "We handled imbalance" without
a number is not an engineering claim, and the only way to have the number is to
be able to run the other arms.

## 2026-08-23, Three seeds, and the standard deviation is reported

**Chosen:** the final configuration runs three times with seeds from
`config.yaml`, and the README reports mean and standard deviation.

**Why:** on a five-class task a single run's macro F1 moves by one to two
points between seeds, which is larger than most differences people publish as
improvements. A single number is an anecdote with a decimal point.

Everything that can be seeded is seeded. Python's `random`, numpy, torch,
CUDA, and the `Trainer`'s own `seed` and `data_seed`, because missing one
makes a run *nearly* reproducible, which is worse than obviously not, since it
looks fine until the numbers move.

## 2026-08-23: The training-set cap is applied after splitting, never before

**Chosen:** split into 70/15/15 first, then cap the **training set** at 60,000
rows.

**Rejected:** subsampling the corpus and then splitting it.

**Why:** capping first shrinks validation and test as well, and every reported
number becomes partly a measurement of the cap. Validation and test always
cover the full 15% of the corpus, 22,159 rows each, so the metrics are
comparable to a run with no cap at all. `test_the_train_cap_does_not_shrink_the_held_out_sets`
asserts it, and `test_the_capped_train_set_is_still_stratified` asserts the cap
takes a stratified slice rather than a head, which would follow whatever order
the upload happened to be in.

## 2026-08-23: The test set is read once, and reading it requires a flag

**Chosen:** `scripts/final_eval.py` is the only code path that reads
`data/splits/test.parquet`, it refuses to run without
`--i-know-this-touches-the-test-set`, and `load_test()` logs a warning every
time it is called.

**Why the flag is not bureaucracy:** a test set read casually during
development is a test set you have tuned against, and every number from it
afterwards is optimistic by an unknown amount. The unknown is the problem: you
cannot correct for it, you cannot report it, and you usually cannot even
remember how many times you looked.

The README states the count.

## 2026-08-23. Handles and URLs are replaced, not deleted

**Chosen:** `@user` becomes `@مستخدم`, URLs become `رابط`.

**Why:** `@AlmFaisal` carries no dialect signal and a great deal of vocabulary.
Left in, a TF-IDF model learns which accounts tweet which dialect, a leak
that inflates validation and evaporates the moment the service sees traffic
from anyone else. Replacing rather than deleting keeps the sentence shape, so
the token count and the n-gram windows stay honest.

The `#` is dropped and the hashtag word kept, because `#طال_عمره` **is**
signal.

## 2026-08-23: One preprocessing function, imported by both training and serving

**Chosen:** `data/preprocess.py:clean` is called by the training pipeline and
by `serving/predictor.py`.

**Why:** training-time and serving-time preprocessing drifting apart is the
single most common way a deployed classifier quietly loses accuracy, and it is
undetectable from either side alone: the model is fine, the service is fine,
the pair is broken. There is exactly one function, and the service imports it
rather than reimplementing the "obvious" parts.

## 2026-08-23: The tokenizer is saved beside the weights, and the pair is asserted at startup

**Chosen:** `trainer.save_model()` and `tokenizer.save_pretrained()` write to
the same directory; `Predictor` refuses to start if the tokenizer is missing,
if its vocabulary is larger than the model's, or if the saved `id2label`
disagrees with `config.yaml`.

**Why:** a tokenizer mismatch does not raise. It produces token ids the model
was never trained on, and the service returns confident nonsense, with high
confidence, because softmax over garbage logits is still a distribution. The
checks are three lines and they turn a silent failure into a refusal to boot.

## 2026-08-23. Micro-batching with a thread, not asyncio

**Chosen:** one background thread with a queue; requests are collected for up
to 10 ms or 32 items, whichever comes first.

**Rejected:** an executor inside the event loop; no batching at all.

**Why batching:** a transformer forward pass on one text and on thirty-two
costs almost the same wall time: the GPU is idle either way, waiting on
kernel launches. One pass per request wastes most of the throughput on light
load and collapses under heavy load.

**Why a thread:** the model call is a blocking, GIL-releasing C extension, so a
thread is both correct and far easier to reason about than the asyncio
equivalent. The added latency is bounded by `max_delay_ms` and nothing else,
which is why that number is small and in `config.yaml` rather than tuned by
feel.

**The bug this design is tested against** is fan-out crossing the wires,
caller A receiving caller B's results. `test_concurrent_callers_each_get_their_own_results`
runs twenty concurrent callers whose texts identify them, so a crossed wire
fails the assertion rather than passing on a coincidence of shapes.

## 2026-08-23. Predictions are logged as a hash, never as text

**Chosen:** the request log stores `sha256(text)[:16]`, character count, word
count, predicted label and confidence.

**Rejected:** logging the raw text, which would enable a stronger
embedding-based drift check.

**Why:** writing user text into a file that outlives the request is a privacy
decision nobody should make by default, and the four signals worth watching,
length, word count, class mix, confidence, do not need it. The hash still
identifies duplicates and still joins a log line back to a request during an
incident.

**What this gives up, stated plainly:** semantic drift that does not change
length or confidence will not be caught. That is a real limitation of this
design, not an oversight.

## 2026-08-23. Evidently is pinned to the 0.7 line, and the docstring says why

**Chosen:** `evidently>=0.7.0,<0.8.0`, and the monitor is written against
`Report` + `Dataset` + `DataDefinition`.

**Why the pin and the warning:** 0.7 rewrote the API. Essentially every
tutorial online, and the archived documentation, use
`evidently.report.Report` with `ColumnDriftMetric`, which does not import on
this version. The API was verified against the installed package before a line
of the monitor was written, and the flattening code reads Evidently's own
per-column `threshold` out of the result rather than hardcoding one here.

**What is compared:** text length, word count, predicted class distribution,
and mean confidence. Confidence is the leading indicator: a model asked about
something it has never seen gets *less certain* before it gets wrong.

**The monitor is tested in both directions.** It must fire when the
distribution really moves and it must **not** fire on a resample of the same
distribution. A monitor that always alerts gets muted within a week, which is
the same as not having one.

## 2026-08-23. MLflow uses a SQLite backend, not `./mlruns`

**Chosen:** `sqlite:///mlflow.db`.

**Why:** MLflow put the filesystem store into maintenance mode and now raises
on `file:./mlruns` unless an opt-out environment variable is set. A database
backend is the supported path, costs nothing locally, and the exception message
that surfaced this was clear, which is worth recording, because the obvious
"fix" is to set `MLFLOW_ALLOW_FILE_STORE=true` and carry on into a deprecated
store.

## 2026-08-23, torch was silently installed as the CPU build, twice

**What happened:** `uv pip install -e .` resolved `torch>=2.4.0` to
`2.13.0+cpu`. Nothing failed. `Trainer` would have trained on the CPU at
roughly a fiftieth of the speed and the only symptom would have been that the
run was slow.

**The fix:** the CUDA wheel is installed explicitly from PyTorch's own index,
as a separate step in `make setup`, and the training script prints
`torch.cuda.is_available()` before it starts.

**Why this is in DECISIONS.md rather than a note:** this is the second time in
this project that a routine `pip install` replaced a CUDA build with a CPU one
(repo 02 hit it too), and both times it was silent. The lesson is that
`torch.cuda.is_available()` belongs in the startup log of anything that expects
a GPU.

## 2026-08-23. CI gates on what CI can actually check

**Chosen:** lint, format, mypy, unit tests, a **training smoke test**, and a
Docker image build.

**The smoke test is the substantive one:** 200 rows, 30 steps, a real
MARBERTv2, on CPU, asserting the training loss falls below ln(5) = 1.61, the
random-guess baseline for five classes. It catches a broken collator, a label
mismatch, a frozen model or a loss that is not connected to the graph, none of
which unit tests see, and it needs no GPU.

**What CI does not do:** the full three-seed training run, the latency
benchmark, or the drift check against real traffic. Those need hardware or a
running service, and faking them in CI would produce a green tick that means
nothing.

## 2026-08-23: The serving image was 3.75 GB because the package had one dependency set

**What happened:** the image built on the first try, which is why the size went
unnoticed for a while: **3.75 GB** for a service that answers `POST /predict`.

**The cause:** `pyproject.toml` had a single `dependencies` list, so
`pip install -e .` inside the container installed `datasets`, `mlflow`,
`matplotlib`, `scikit-learn`, `scipy` and `joblib`: an experiment tracker, a
plotting library and a training data loader, shipped into production.

**The fix:** three sets.

| Set | Holds | Installed by |
|---|---|---|
| base | fastapi, transformers, optimum, onnxruntime, pandas | the container |
| `train` | datasets, torch, mlflow, matplotlib, scikit-learn, scipy | `make setup`, CI |
| `monitor` | evidently | the drift job |

**Result: 3.75 GB to 2.43 GB, a 35% cut**, and the container was re-run to
confirm it still serves: identical labels and confidences to three decimal
places against the GPU PyTorch service on the same five hand-written sentences.

**Why it is still 2.43 GB, stated rather than hidden.** `optimum[onnxruntime]`
depends on `torch`, so the CPU torch wheel arrives transitively even though the
ONNX backend never calls it. Getting rid of it means dropping optimum and
driving `onnxruntime` directly with a `transformers` tokenizer, a real
refactor, worth doing if image size matters, and not done here. The CPU wheel
is installed explicitly from PyTorch's index first so that at least the 2.5 GB
CUDA build does not arrive instead.

**A gap the same exercise found.** `/model-info` reported
`validation_macro_f1: null` from the ONNX directory, because the export copied
the tokenizer and the config but not `training_record.json`. An endpoint that
offers a field and answers `null` is worse than one that does not offer it.
`export()` now copies the record, the same way `quantize()` already did.
