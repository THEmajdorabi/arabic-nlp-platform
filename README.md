# arabic-nlp-platform

Arabic dialect identification — Egyptian, Libyan, Lebanese, Sudanese, Moroccan
— from a dataset nobody had cleaned to a monitored service with a drift alert
that has actually fired.

The point of this repo is not the F1. It is that every number below has a
baseline next to it, an interval around it, and a command that regenerates it.

```bash
make data       # download, clean, stratified split, class distribution report
make baseline   # TF-IDF + logistic regression. Under a minute of CPU
make train      # MARBERTv2, three seeds. About an hour on one GPU
make export     # ONNX + int8, with the accuracy re-checked
make serve      # the API on :8000
make simulate-drift   # make the alert fire, on purpose
```

---

## Results

**Everything here is the test set, read once, at the end.** Validation numbers
live in `artifacts/`; this section is the only place the test split appears.

| | TF-IDF + logistic regression | MARBERTv2 fine-tuned |
|---|---|---|
| **macro F1** | 0.8166 | **0.8658** [0.8602, 0.8709] |
| weighted F1 | 0.8458 | 0.8905 |
| expected calibration error | **0.0346** | 0.0666 |
| training cost | 53 s, CPU | 3 x 17 min, one RTX 3060 Ti |

**Paired over the same 22,159 test examples: +0.0493 macro F1, 95% CI
[+0.0437, +0.0550].** The interval excludes zero, so the difference is
real — and paired is the test that can say so, because the two unpaired
intervals would leave it arguable.

**Three seeds on validation: macro F1 0.8741 ± 0.0009.** A standard deviation
of 0.0009 across seeds is smaller than most published improvements, which is
the point of running three: without it, a single run's number carries an
unknown amount of seed luck.

**The baseline is better calibrated.** ECE 0.0346 against 0.0666 — the
transformer is five points more accurate and nearly twice as overconfident. For
a classifier feeding a routing decision that matters: `/route` uses a 0.60
confidence floor, and a model whose 0.9 means 0.83 clears that floor more often
than it should. Neither model is temperature-scaled; that is listed under
*Not done*.

**The deployed model is seed 20260823 — the first seed, not the best of three.**
The spread is 0.0009. Picking the best of three on validation is a selection
bias for a difference that small, and `config.yaml` lists the seeds in order so
the choice is reproducible rather than a judgement call.

### Per class

Test split, sorted by support — the order that makes the pattern visible.

| Class | Support | Baseline F1 | Fine-tuned F1 | Precision | Recall |
|---|---|---|---|---|---|
| Egyptian (EG) | 8,645 | 0.892 | **0.931** | 0.921 | 0.940 |
| Libyan (LY) | 5,475 | 0.830 | **0.873** | 0.884 | 0.862 |
| Lebanese (LB) | 4,143 | 0.866 | **0.909** | 0.901 | 0.917 |
| Sudanese (SD) | 2,165 | 0.720 | **0.792** | 0.806 | 0.778 |
| Moroccan (MA) | 1,731 | 0.776 | **0.825** | 0.836 | 0.815 |

**The gap tracks class size almost exactly.** Egyptian, at 39% of the corpus,
scores 0.931; Sudanese, at 9.8%, scores 0.792. Class weights in the loss
narrow that gap and do not close it, and the honest reading is that
2,165 Sudanese test rows is not much to learn a dialect that shares most of
its vocabulary with the largest class.

Across the three seeds every per-class standard deviation is under 0.004 — the
largest is Moroccan at 0.0038 — so this ranking is not seed noise.

### What the errors look like

The 30 most **confident** mistakes are read line by line in
[docs/error_analysis.md](docs/error_analysis.md). The headline from it:

**Twelve of the thirty carry the unmistakable markers of the class the model
predicted, and none of the class they are labelled with.** Checked mechanically
against the baseline's own top-weighted features, so the criterion comes from
the data rather than from my reading. Two examples:

- *"وانا كنحماااق عليكم يا اهل المغرب"* — labelled **LY**, uses the Moroccan
  progressive `كن-`, and says it is addressed to Moroccans. Predicted MA at
  0.999.
- *"وين جايب هالحكي انت"* — labelled **EG**; `وين` and `هالحكي` are Levantine.
  Predicted LB at 0.998.

Four of the thirty are genuine model errors. The remaining fourteen carry no
marker either way — they average 50.6 characters against a corpus mean of 87.5,
which is the other structural cause: short mention-led replies with emoji and
very little text left after handles are replaced.

**What this means for the ceiling.** If a third of the most confident errors are
mislabelled rows, macro F1 here is bounded above by label quality and the last
few points are not available to any model. What it does *not* mean is that a
third of the corpus is mislabelled — these thirty are the tail of the confidence
distribution, deliberately selected to be the model's worst beliefs, and
mislabelled rows are exactly what accumulates there. Estimating the real rate
needs a random sample read by a person, which has not been done.

### Latency

Same machine, same run: NVIDIA GeForce RTX 3060 Ti and an AMD Zen 3 CPU.
**Three rows, because only two of them are a fair comparison** — ONNX Runtime
here is CPU-only, so "ONNX is slower than PyTorch" measured against PyTorch on a
GPU would be a statement about devices, not runtimes.

| Runtime | Batch | p50 ms | p95 ms | p99 ms | items/s |
|---|---|---|---|---|---|
| pytorch-cuda | 1 | 8.1 | 8.6 | 9.01 | 134.0 |
| pytorch-cuda | 8 | 14.55 | 16.66 | 16.89 | 542.1 |
| pytorch-cuda | 32 | 49.79 | 50.95 | 51.56 | 645.2 |
| pytorch-cpu | 1 | 41.97 | 50.51 | 53.63 | 23.4 |
| pytorch-cpu | 8 | 283.4 | 300.99 | 308.89 | 28.1 |
| pytorch-cpu | 32 | 1050.03 | 1072.54 | 1122.8 | 30.4 |
| onnx-cpu | 1 | 21.41 | 22.59 | 24.73 | 46.3 |
| onnx-cpu | 8 | 252.47 | 262.06 | 263.83 | 31.6 |
| onnx-cpu | 32 | 981.77 | 1016.68 | 1057.67 | 32.5 |

**ONNX Runtime nearly halves single-item CPU latency**: 21.41 ms against
41.97 ms, a 1.96x speedup at batch 1. The advantage shrinks as the batch
grows — by batch 32 it is 1.07x — because the graph optimisations that help a
small kernel launch matter less once the matrices are large.

**The GPU beats both by a wide margin**, 8.1 ms at batch 1. If a GPU is
available, use it; the CPU path exists so the container runs anywhere.

### Quantization: measured, and rejected

| | PyTorch | ONNX fp32 | ONNX int8 |
|---|---|---|---|
| macro F1 (4,000 validation rows) | 0.8892 | 0.8892 | 0.8743 |
| size on disk | — | 655.4 MB | 167.4 MB |

**Exporting is lossless.** PyTorch and ONNX fp32 agree to four decimal places
(+0.0000) — the check that says the traced graph is the model that was
evaluated rather than something resembling it.

**Quantizing is not.** int8 costs **0.0149 macro F1** for a 3.9x smaller file.
The budget in `scripts/export_onnx.py` is 0.010, so the export report
recommends fp32 and `docker-compose.yml` mounts the fp32 graph. The int8 model
is still built; the point is that it was checked, and the check said no.

---

## The data

147,725 Arabic tweets, five country dialects, MIT licence, downloaded
2026-08-23. **39.0% Egyptian, 7.8% Moroccan — a 4.99x imbalance**, which is why
accuracy appears nowhere in this repo and macro F1 with per-class breakdowns
appears everywhere.

[data/README.md](data/README.md) records the choice, the three datasets
rejected and why, the cleaning, and the split recipe. The short version of the
rejections: **QADI has no text** (540k tweet IDs needing rehydration), and the
440k-row alternative has **18 integer classes with no label map and no
declared licence** — per-class F1 for a class whose name is a guess is not a
result.

## Why the baseline comes first

A portfolio that reports "0.87 macro F1" says nothing. Reported against a
baseline, it says what the GPU hour bought.

**TF-IDF over character 3-5 grams plus word unigrams, into logistic regression
with balanced class weights: macro F1 0.8203, in
53 seconds on a CPU.** That is the number the
transformer has to beat.

Character n-grams are the substantive choice, not a default. Arabic dialect
markers are frequently sub-word -- the Levantine future prefix `ح-`, the
Egyptian progressive `ب-`, the Maghrebi negation circumfix `ما...ش`. A
word-unigram model sees `حيروح` and `رايح` as unrelated tokens; a character
model sees the shared marker.

**Its top-weighted features are the cheapest sanity check that exists**, and
they came out as textbook dialect markers:

| Class | Highest-weighted features |
|---|---|
| LB | `عم` `شو` `هيك` `رح` `هيدا` `حدا` `بدو` |
| MA | `هاد` `واش` `ديال` `باش` `حيت` `راه` `حنا` |
| SD | `زول` `ياخ` `ديل` `منو` `دا` |
| LY | `شن` `توا` `هكي` `هضا` `ليبيا` |
| EG | `ده` `دي` `علشان` `كام` `ربنا` |

If those had come out as usernames or punctuation, the labels would be wrong
and no amount of F1 would have said so.

**A result from the ablation, which contradicted the guess.** Arabic
normalization -- folding hamza, tashkeel and alef -- *hurts* here:
+0.0041 macro F1 for turning it off, 95% CI
[+0.0009, +0.0074], paired over the same
22,159 validation examples. The two unpaired intervals overlap
almost completely and the first conclusion written was "not a finding"; the
paired test is what settled it. In dialect identification the spelling
variation *is* the signal, which is the opposite of repo 02, where the same
folding helps. See DECISIONS.md.

## The test set

Read **once**, at the end, by `scripts/final_eval.py` — which refuses to run
without `--i-know-this-touches-the-test-set`. Every other number in this repo
is from the validation split.

**Count of times the test set has been read: 1.** Every model decision above —
the imbalance strategy, the number of epochs, whether to normalize Arabic,
which seed to deploy, whether to ship int8 — was made on validation, before
that single read.

A test set read casually during development is one you have tuned against, and
every number from it afterwards is optimistic by an unknown amount. The unknown
is the problem: you cannot correct for it and you usually cannot remember how
many times you looked.

## Serving

```
POST /predict     {texts: [...]}  -> [{label, confidence, all_scores}]
POST /route       one question    -> a routing decision, or an honest "unknown"
GET  /healthz
GET  /model-info  base model, seed, imbalance strategy, validation macro F1
GET  /metrics     Prometheus
GET  /drift       the current drift report
```

**Micro-batching.** A transformer forward pass on one text and on thirty-two
costs nearly the same wall time — the GPU is idle either way, waiting on kernel
launches. Requests are collected for up to 10 ms or 32 items, whichever comes
first, then run as one pass.

**Measured, batch 1 against batch 32 on the GPU: 134.0 items/s to
645.2 items/s, a 4.8x throughput gain** for at most 10 ms of added latency.

On the CPU path the gain is negative — 46.3 items/s at batch 1 falls to
32.5 at batch 32 — because a CPU forward pass was never idle waiting for
kernel launches in the first place. The batcher is still correct there; it is
just not a win, and that is worth knowing before deploying it on CPU nodes and
expecting throughput.

**`/route` says "unknown" rather than guessing.** Below a confidence floor it
returns `route: "default"` and no dialect. A confident wrong route is worse
than no route, and repo 02's client honours that verdict rather than reading
past it to the raw label.

**Predictions are logged as a hash of the text, never the text.** Length, word
count, predicted label and confidence are enough for every drift signal this
repo watches, and writing user text into a file that outlives the request is a
privacy decision nobody should make by default. What that gives up: semantic
drift that changes neither length nor confidence will not be caught.

## Monitoring and drift

Prometheus scrapes the service; Grafana is provisioned as code in
`grafana/dashboards/` — request rate, p50/p95/p99, error rate, class
distribution over time, mean confidence, and the drift gauge.

Four things are compared against a training-time reference: **text length**,
**word count**, **predicted class distribution**, and **mean confidence**.
Confidence is the leading indicator — a model asked about something it has
never seen gets *less certain* before it gets wrong.

**The alert has fired, and this is the run.** `scripts/simulate_drift.py` feeds
the service formal MSA — a register with no dialect markers at all, which the
model has never been trained on — and then the scheduled check runs.

The negative control first, 400 held-out validation texts:

```
DRIFT  0/4 columns (share 0.00, threshold 0.50)
  chars       Wasserstein     0.0990   no
  words       Wasserstein     0.0889   no
  confidence  Wasserstein     0.0608   no
  label       Jensen-Shannon  0.0402   no
```

Then 400 MSA texts:

```
DRIFT  4/4 columns (share 1.00, threshold 0.50)   *** ALERT ***
  chars       Wasserstein     0.6453   yes
  words       Wasserstein     0.6810   yes
  confidence  Wasserstein     2.5962   yes
  label       Jensen-Shannon  0.3550   yes
```

`check_drift.py` exits **1** on the alert, so a scheduler or a CI job can act on
it rather than only writing a file. The saved report is
[docs/drift_alert_example.json](docs/drift_alert_example.json).

**Confidence moved four times further than any other column**, which is the whole
argument for monitoring it: the model got *less certain* before it got wrong.

**A bug this measurement caught.** The reference was originally built from the
**training** split, and the very first check against ordinary validation traffic
flagged `confidence` as drifted — because the model is systematically more
confident on text it was fitted to. That is a property of the reference, not of
the traffic, and it would have burned a quarter of the alert budget on every
window forever. The reference is now built from the held-out validation split.
A monitor that always alerts gets muted within a week, which is the same as not
having one.

**Retraining policy**, written down because automated retraining without a gate
is how models silently degrade:

| | |
|---|---|
| **Trigger** | `dialect_drift_share >= 0.5` sustained for 10 minutes, or mean confidence below 0.55 for 15 minutes. |
| **Window** | The last 500 logged predictions against a 5,000-row training reference. |
| **First action** | *Not* a retrain. Pull the drifting column's distribution and find out what changed — a new client, a new language, a bot, or a genuine shift in who is asking. |
| **Retrain when** | The shift is genuine and persistent across two consecutive daily windows. |
| **Who approves** | A human. The pipeline produces a candidate model and its full evaluation; nothing is promoted on a metric alone. |
| **Promotion gate** | The candidate must beat the incumbent on macro F1 **on the held-out test set, paired**, with an interval that excludes zero. A point-estimate improvement is not enough. |

## Reproducing it

```bash
make setup     # note: installs the CUDA torch build explicitly. See below.
dvc repro      # every stage, from download to latency benchmark
```

`dvc.yaml` covers download → clean → split → baseline → transformer → ONNX →
latency, with `config.yaml` as the parameter file, so changing a hyperparameter
invalidates exactly the stages that depend on it.

**One trap worth naming.** `uv pip install -e .` resolves `torch>=2.4.0` to the
**CPU** wheel, silently. Nothing fails; training just runs about fifty times
slower and the only symptom is that it is slow. `make setup` installs the CUDA
build explicitly, and the training script prints `torch.cuda.is_available()`
before it starts. This happened twice in this project — repo 02 hit it too.

## CI

Lint, format, mypy, unit tests, a **training smoke test**, and a Docker build.

The smoke test is the substantive one: 200 rows, 30 steps, a real MARBERTv2 on
CPU, asserting the training loss falls below ln(5) = 1.61 — the random-guess
baseline for five classes. It catches a broken collator, a label mismatch, a
frozen model, or a loss disconnected from the graph, none of which unit tests
see, and it needs no GPU.

CI does **not** run the full training, the latency benchmark, or a drift check
against real traffic. Those need hardware or a running service, and faking them
would produce a green tick that means nothing.

## Design notes

Every real choice is in [DECISIONS.md](DECISIONS.md) with the evidence,
including the one that contradicted the guess written in the config:

- **Arabic normalization slightly hurts here** (−0.004 macro F1 on the
  baseline), the opposite of repo 02, where it helps. In dialect ID the
  spelling variation *is* signal;
- the tokenizer is saved beside the weights and the pair is asserted at
  startup, because a tokenizer mismatch does not raise — it returns confident
  nonsense;
- Evidently is pinned to the 0.7 line, and the API was verified against the
  installed package before a line was written, because every tutorial online
  uses the 0.4-era API that does not import;
- the training-set cap is applied *after* splitting, so validation and test
  always cover the full held-out data.

## Not done

- **A human has not read a random sample of the labels.** The error analysis
  says a third of the model's most confident mistakes look mislabelled; turning
  that into a label-error rate needs a person and 200 random rows. Every number
  above is bounded by that unknown.
- **Neither model is temperature-scaled.** The transformer is the more accurate
  and the less calibrated of the two (ECE 0.0666 against 0.0346), so
  `/route`'s confidence floor is doing less work than it appears to.
  Temperature scaling on the validation split is an hour's work and is the
  highest-value thing left undone.
- **Only the `class_weights` imbalance strategy was run to three seeds.**
  `oversample` and `none` are implemented and selectable from `config.yaml`;
  they have not been measured, so no claim is made about what class weighting
  was worth.
- **The Grafana dashboard is provisioned but never rendered.** It is defined as
  code in `grafana/dashboards/dialect-service.json` and every metric it queries
  is live on `/metrics`; nobody has looked at it in a browser.
- **The 60,000-row training cap has not been lifted.** Whether the remaining
  43,000 training rows are worth anything is unmeasured.
- **Repo 02 does not act on the routing decision.** The hook is wired end to end
  and the dialect is reported; whether routing on it improves retrieval has not
  been A/B tested, and until it has, it is an integration rather than a
  feature.
