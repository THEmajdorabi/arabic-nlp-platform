"""Export the best model to ONNX, quantize it, and verify accuracy.

python scripts/export_onnx.py --model-dir artifacts/transformer/class_weights_seed20260823/model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from arabic_nlp.config import load_settings  # noqa: E402
from arabic_nlp.data.splits import load_split  # noqa: E402
from arabic_nlp.export_onnx import export_and_verify  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", default="artifacts/onnx")
    parser.add_argument("--rows", type=int, default=4000, help="validation rows used to verify")
    parser.add_argument("--max-drop", type=float, default=0.01)
    args = parser.parse_args()

    settings = load_settings()
    labels = settings.dataset.label_names
    index = {name: i for i, name in enumerate(labels)}

    validation = load_split("validation").head(args.rows)
    texts = validation["text"].tolist()
    y_true = np.array([index[label] for label in validation["label"]])

    result = export_and_verify(
        args.model_dir, ROOT / args.out, texts, y_true, settings, args.max_drop
    )

    print("ONNX EXPORT")
    print("| | fp32 | int8 |")
    print("|---|---|---|")
    print(
        f"| macro F1 (n={len(texts)}) | {result.fp32_macro_f1:.4f} | "
        f"{f'{result.int8_macro_f1:.4f}' if result.int8_macro_f1 is not None else 'n/a'} |"
    )
    print(
        f"| size on disk | {result.fp32_size_mb:.1f} MB | "
        f"{f'{result.int8_size_mb:.1f} MB' if result.int8_size_mb else 'n/a'} |"
    )
    if result.delta is not None:
        print()
        print(f"quantization changed macro F1 by {result.delta:+.4f}")
    print(f"recommended: {result.recommended}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
