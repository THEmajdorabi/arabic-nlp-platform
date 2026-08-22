.PHONY: setup data baseline train export latency serve drift simulate-drift test lint format typecheck smoke final clean

PY := .venv/Scripts/python.exe
MODEL := artifacts/transformer/class_weights_seed20260823/model

setup:  ## venv and dependencies. Installs the CUDA torch build explicitly.
	uv venv --python 3.12
	uv pip install -e .
	uv pip install pytest ruff mypy types-PyYAML dvc
	uv pip install torch --index-url https://download.pytorch.org/whl/cu124 --reinstall

data:  ## download, clean, split, report the class distribution
	$(PY) scripts/prepare_data.py

baseline:  ## TF-IDF + logistic regression. CPU, a few minutes
	$(PY) scripts/train_baseline.py

train:  ## MARBERTv2, three seeds. GPU, about an hour
	$(PY) scripts/train_transformer.py --seeds 3

smoke:  ## the CI training test: 200 rows, asserts the loss falls
	$(PY) scripts/train_transformer.py --smoke

export:  ## ONNX export, int8 quantization, accuracy re-check
	$(PY) scripts/export_onnx.py --model-dir $(MODEL)

latency:  ## p50/p95/p99 for PyTorch and ONNX, same machine
	$(PY) scripts/benchmark_latency.py --torch-dir $(MODEL) --onnx-dir artifacts/onnx/fp32

serve:  ## the API on http://127.0.0.1:8000
	MODEL_DIR=$(MODEL) $(PY) -m uvicorn arabic_nlp.serving.app:app --host 127.0.0.1 --port 8000

drift:  ## compare the request log against the reference
	$(PY) scripts/check_drift.py

reference:  ## build the drift reference. Run once, after training
	$(PY) scripts/check_drift.py --build-reference --model-dir $(MODEL)

simulate-drift:  ## make the alert fire, on purpose
	$(PY) scripts/simulate_drift.py --mode msa
	$(PY) scripts/check_drift.py

final:  ## score the TEST set. Once. At the end.
	$(PY) scripts/final_eval.py --model-dir $(MODEL) \
		--baseline artifacts/baseline/model_normalized.joblib \
		--i-know-this-touches-the-test-set

test:  ## no GPU, no network, no downloaded dataset
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

typecheck:
	$(PY) -m mypy src/arabic_nlp

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache artifacts mlruns mlartifacts mlflow.db logs
