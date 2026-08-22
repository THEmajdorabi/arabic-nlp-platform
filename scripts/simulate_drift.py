"""Feed the service a deliberately shifted sample so the alert actually fires.

A drift monitor nobody has seen trigger is not evidence of anything. This sends
a batch the model was not trained for and then runs the drift check, so the
alert firing is a reproducible command rather than a claim.

    python scripts/simulate_drift.py --mode short     # much shorter texts
    python scripts/simulate_drift.py --mode msa       # formal MSA, not dialect
    python scripts/simulate_drift.py --mode skew      # one dialect only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402

from arabic_nlp.config import load_settings  # noqa: E402
from arabic_nlp.data.splits import load_split  # noqa: E402

# Formal MSA: the register the model has never been trained on. Every one of
# these is grammatical written Arabic with no dialect markers at all.
MSA_TEXTS = [
    "تشير التقديرات إلى أن معدل النمو الاقتصادي سيرتفع خلال العام المقبل.",
    "أصدرت الوزارة تعميماً يقضي بتنظيم ساعات العمل في القطاع الخاص.",
    "يهدف هذا التقرير إلى تحليل البيانات المتاحة حول سوق العمل.",
    "وافق مجلس الإدارة على الميزانية السنوية بعد مناقشة مستفيضة.",
    "تنص المادة الخامسة على وجوب إخطار الطرف الآخر كتابةً.",
    "أكد المتحدث الرسمي أن الإجراءات ستدخل حيز التنفيذ مطلع الشهر.",
    "تم توقيع مذكرة تفاهم بين الجهتين لتعزيز التعاون المشترك.",
    "أوضح التقرير أن نسبة البطالة انخفضت مقارنة بالربع السابق.",
]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="msa", choices=["msa", "short", "skew"])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--rows", type=int, default=300)
    args = parser.parse_args()

    settings = load_settings()
    validation = load_split("validation")

    if args.mode == "msa":
        texts = [MSA_TEXTS[i % len(MSA_TEXTS)] for i in range(args.rows)]
        note = "formal MSA, a register with no dialect markers"
    elif args.mode == "short":
        texts = [t[:18] for t in validation["text"].tolist()[: args.rows]]
        note = "texts truncated to 18 characters"
    else:
        one = validation[validation["label"] == "MA"]["text"].tolist()
        texts = (one * (args.rows // max(1, len(one)) + 1))[: args.rows]
        note = "a single dialect (MA) instead of the training mix"

    print(f"sending {len(texts)} texts: {note}")
    client = httpx.Client(timeout=60.0)
    sent = 0
    for start in range(0, len(texts), 32):
        batch = texts[start : start + 32]
        response = client.post(f"{args.url}/predict", json={"texts": batch})
        response.raise_for_status()
        sent += len(batch)
    print(f"sent {sent} texts to {args.url}")
    print()
    print("now run:  python scripts/check_drift.py")
    print(f"(the service logs to {settings.serving.request_log})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
