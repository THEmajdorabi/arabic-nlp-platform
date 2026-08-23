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

<!-- HEADLINE -->

### Per class

<!-- PERCLASS -->

### What the errors look like

<!-- ERRORS -->

### Latency

<!-- LATENCY -->

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

<!-- BATCHING -->

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

<!-- DRIFT -->

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

<!-- NOTDONE -->
