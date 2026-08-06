#!/usr/bin/env python3
"""
translate_jargon_dataset.py — ConfigProfiler, jargon-fidelity dataset
(step 1b: generating real on-device translations).

Sends every sentence from the corpus (jargon_dataset_corpus.csv) to
llama_main on the device, with an EN->FR translation instruction, thinking
mode disabled (/no_think, decision settled Day 2/3 — see EXECUTION_LOG.md).
Produces a translations_results.csv file directly compatible with
score_jargon_fidelity.py (columns id, quantization, translation_fr).

Reuses Device and run_on_device_benchmark from collect_benchmarks.py (same
folder) — no duplication of the adb logic already validated on a real device.

USAGE:
    python3 translate_jargon_dataset.py --corpus jargon_dataset_corpus.csv \
        --serial <IP:PORT or serial number> --quantizations q8da4w \
        --threads 4 --out translations_results.csv

⚠️ EXTRACTING THE TRANSLATION FROM THE RAW OUTPUT — a fragile technical
point, explicitly documented:
    llama_main's raw output contains the echoed prompt, a known gluing
    artifact ("assistantius" observed Day 2 — the first generated token
    sometimes glues directly onto the <|im_start|>assistant marker with no
    space, due to a tokenization detail), the <think>...</think> block
    (empty thanks to /no_think, but still present as a tag), then the
    translated text, then <|im_end|>, then the PyTorchObserver JSON line.

    Extraction strategy: cut AFTER the last occurrence of "</think>"
    (reliable, never glued to anything else) rather than after
    "<|im_start|>assistant" (subject to the gluing artifact). Fall back to
    the "assistant" method only if no </think> tag is found (shouldn't
    happen with /no_think active, but we don't want to fail silently if the
    model's behavior ever changes).

⚠️ ABOUT --dry-run — same warning as collect_benchmarks.py: this mode
simulates adb with no connected device (via the imported Device class),
producing NO real generated text — every translation will show up as
"⚠️ EMPTY" in dry-run. This is expected and only validates the pipeline
(resume logic, CSV schema, argument parsing). Never cite a --dry-run output
as a real translation.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from collect_benchmarks import Device, AdbError, run_on_device_benchmark


TRANSLATION_INSTRUCTION_TEMPLATE = (
    "Translate the following English sentence into French. "
    "Reply with only the French translation, nothing else — no quotes, "
    "no explanation.\n\nEnglish: {source_en}"
)


def extract_translation(raw_output: str) -> str:
    """Extracts the translated text from llama_main's raw output.

    See the module docstring for the extraction pitfall detail (the
    "assistantius" artifact) and why we cut after </think>.
    """
    think_close = "</think>"
    idx = raw_output.rfind(think_close)
    if idx != -1:
        tail = raw_output[idx + len(think_close):]
    else:
        # Fallback in case </think> is ever absent (shouldn't happen with
        # /no_think active — see docstring).
        marker = "<|im_start|>assistant"
        idx2 = raw_output.rfind(marker)
        tail = raw_output[idx2 + len(marker):] if idx2 != -1 else raw_output

    for stop in ("<|im_end|>", "PyTorchObserver"):
        pos = tail.find(stop)
        if pos != -1:
            tail = tail[:pos]

    return tail.strip().strip('"').strip()


def load_corpus(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quantizations", nargs="+", default=["q8da4w"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N sentences (for a quick test before the real run).")
    args = parser.parse_args()

    device = Device(serial=args.serial, dry_run=args.dry_run)
    corpus = load_corpus(args.corpus)
    if args.limit:
        corpus = corpus[:args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    is_new = not args.out.exists()

    already_done = set()
    if not is_new:
        with args.out.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                already_done.add((row["id"], row["quantization"]))

    with args.out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "quantization", "threads", "translation_fr", "notes"])
        if is_new:
            writer.writeheader()

        total = len(corpus) * len(args.quantizations)
        done_count = 0
        for quant in args.quantizations:
            for row in corpus:
                rid = row["id"]
                if (rid, quant) in already_done:
                    print(f"[skip - already done] {rid} ({quant})")
                    done_count += 1
                    continue

                prompt = TRANSLATION_INSTRUCTION_TEMPLATE.format(source_en=row["source_en"])
                print(f"[{done_count+1}/{total}] {rid} ({quant}) ...", end=" ", flush=True)
                try:
                    bench = run_on_device_benchmark(device, quant, args.threads, prompt=prompt)
                    translation = extract_translation(bench.raw_output)
                    notes = ""
                    if not translation:
                        notes = "WARNING: empty extraction, inspect raw_output manually"
                        print("⚠️ EMPTY")
                    else:
                        print(f"→ \"{translation[:60]}...\"" if len(translation) > 60 else f"→ \"{translation}\"")
                    writer.writerow({
                        "id": rid, "quantization": quant, "threads": args.threads,
                        "translation_fr": translation, "notes": notes,
                    })
                    f.flush()
                except AdbError as e:
                    print(f"❌ ERROR: {e}", file=sys.stderr)
                    writer.writerow({
                        "id": rid, "quantization": quant, "threads": args.threads,
                        "translation_fr": "", "notes": f"ERROR: {e}",
                    })
                    f.flush()
                done_count += 1

    print(f"\nTranslations written to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
