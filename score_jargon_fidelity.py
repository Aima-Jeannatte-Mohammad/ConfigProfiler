#!/usr/bin/env python3
"""
score_jargon_fidelity.py — ConfigProfiler, jargon-fidelity dataset (step 2)

Computes the technical-jargon preservation rate in EN->FR translations, via
automatic matching against a canonical-term glossary (method settled Day 3
— see EXECUTION_LOG.md: objective, reproducible, not a subjective judgment
call at read time).

METHOD:
    For every sentence in the "jargon" category, the corpus defines a list
    of expected canonical terms (jargon_terms, separated by ';'). A
    translation is considered "preserved" for a term if that term appears
    as-is (substring search, case-insensitive and tolerant to simple
    accents) in the translated text. A run is "preserved" overall if ALL of
    its canonical terms are found.

    Sentences in the "control" category are not scored for jargon (no term
    expected) — they serve as a qualitative comparison (length, general
    coherence) but not a quantitative measure of technical fidelity.

    SAFEGUARD ADDED DAY 3 (methodological question raised mid-project — see
    EXECUTION_LOG.md): "jargon preserved" must NOT be confused with "the
    model didn't translate anything at all" (e.g. a pure copy of the English
    sentence, a total translation failure under severe quantization — a
    scenario where the preservation score would be artificially perfect
    while the translation is a complete failure). Every row therefore also
    gets a `looks_translated` flag: presence of at least one common French
    function word (articles, prepositions...) in the text. A row with
    `all_preserved=True` BUT `looks_translated=False` is a strong warning
    sign — probably a pure copy, not genuine intelligent jargon preservation
    within a real translation.

⚠️ WHAT THIS SCRIPT DOES NOT DO:
    - Does not judge grammatical quality/fluency beyond the `looks_translated`
      safeguard above (binary, not a quality score) — "control" rows serve
      as an at-minimum qualitative comparison, not an automated score.
    - An exact substring match can produce a FALSE NEGATIVE if the term is
      legitimately rephrased (e.g. "quantized" -> "quantifié" in French,
      which is a correct translation of the concept but doesn't match the
      canonical string "quantization"). A quick manual review of the "not
      preserved" cases remains necessary before publishing a final number —
      this script gives a reproducible first-level signal, not a 100%
      automated final verdict.

USAGE:
    python3 score_jargon_fidelity.py --corpus jargon_dataset_corpus.csv \
        --results translations_results.csv --out scoring_report.csv

The --results file must contain at least these columns:
    id, quantization, translation_fr
(threads, cache_state, or any other context column are kept as-is in the
report if present, but are not required.)

⚠️ THIS SCRIPT HAS NO NOTION OF "SIMULATED" DATA — it scores whatever CSV it
is given. It is the CALLER's responsibility to ensure --results was produced
by translate_jargon_dataset.py WITHOUT --dry-run (see that script's own
warning). Scoring a --dry-run output here would silently produce a nonsense
0% preservation rate (dry-run translations are always empty) rather than a
signal about real model behavior.
"""

from __future__ import annotations

import argparse
import csv
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path


def strip_accents(s: str) -> str:
    """Normalizes accents for a more tolerant comparison (e.g. if a
    canonical term contains an accent that might vary in the output)."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def normalize(s: str) -> str:
    return strip_accents(s).lower()


# Very common French function words, almost impossible to avoid in a real
# French sentence, even a short one. Serves as a minimal safeguard against a
# pure copy of the English source (see module docstring).
_FRENCH_FUNCTION_WORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "est", "sont",
    "avec", "pour", "dans", "sur", "nous", "notre", "nos", "au", "aux",
    "ce", "cette", "ces", "que", "qui", "a", "ont", "se", "son", "sa", "ses",
}


def looks_translated(text_fr: str) -> bool:
    """Minimal heuristic: at least one common French function word present
    in the text. Doesn't prove translation quality, but catches the crude
    case of a pure copy of the English source (see safeguard documented at
    the top of this file)."""
    words = set(normalize(text_fr).split())
    return bool(words & _FRENCH_FUNCTION_WORDS)


def load_corpus(path: Path) -> dict[str, dict]:
    corpus = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            terms = [t.strip() for t in row["jargon_terms"].split(";") if t.strip()]
            corpus[row["id"]] = {
                "category": row["category"],
                "source_en": row["source_en"],
                "jargon_terms": terms,
                "notes": row.get("notes", ""),
            }
    return corpus


def score_translation(translation_fr: str, jargon_terms: list[str]) -> dict:
    """Returns term-by-term detail plus the overall status for one translation."""
    norm_translation = normalize(translation_fr)
    term_results = {}
    for term in jargon_terms:
        term_results[term] = normalize(term) in norm_translation
    all_preserved = all(term_results.values()) if term_results else None
    return {"term_results": term_results, "all_preserved": all_preserved}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)

    detailed_rows = []
    missing_terms_log = []  # for manual review
    suspect_copy_log = []   # preserved but likely a pure copy (safeguard)
    stats_by_quant = defaultdict(lambda: {"n": 0, "preserved": 0, "suspect_copy": 0})

    with args.results.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        result_rows = list(reader)

    if not result_rows:
        print("[error] --results file is empty.", file=sys.stderr)
        sys.exit(1)

    for row in result_rows:
        rid = row["id"]
        quant = row.get("quantization", "unknown")
        translation = row.get("translation_fr", "")

        if rid not in corpus:
            print(f"[warning] id '{rid}' not found in corpus, row skipped.", file=sys.stderr)
            continue

        entry = corpus[rid]

        if entry["category"] == "control":
            detailed_rows.append({
                "id": rid, "category": "control", "quantization": quant,
                "source_en": entry["source_en"], "translation_fr": translation,
                "jargon_terms": "", "all_preserved": "", "terms_missing": "",
                "looks_translated": looks_translated(translation),
            })
            continue

        result = score_translation(translation, entry["jargon_terms"])
        missing = [t for t, ok in result["term_results"].items() if not ok]
        translated = looks_translated(translation)

        detailed_rows.append({
            "id": rid, "category": "jargon", "quantization": quant,
            "source_en": entry["source_en"], "translation_fr": translation,
            "jargon_terms": ";".join(entry["jargon_terms"]),
            "all_preserved": result["all_preserved"],
            "terms_missing": ";".join(missing),
            "looks_translated": translated,
        })

        stats_by_quant[quant]["n"] += 1
        if result["all_preserved"]:
            stats_by_quant[quant]["preserved"] += 1
        if not translated:
            stats_by_quant[quant]["suspect_copy"] += 1
        if missing:
            missing_terms_log.append((rid, quant, missing, translation))
        if result["all_preserved"] and not translated:
            suspect_copy_log.append((rid, quant, translation))

    # Write the detailed report
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "category", "quantization", "source_en", "translation_fr",
            "jargon_terms", "all_preserved", "terms_missing", "looks_translated",
        ])
        writer.writeheader()
        writer.writerows(detailed_rows)

    # Console summary
    print(f"\nDetailed report written to: {args.out.resolve()}\n")
    print("=== Jargon preservation rate by quantization ===")
    for quant, s in sorted(stats_by_quant.items()):
        rate = 100.0 * s["preserved"] / s["n"] if s["n"] else 0.0
        copy_rate = 100.0 * s["suspect_copy"] / s["n"] if s["n"] else 0.0
        print(f"  {quant}: {s['preserved']}/{s['n']} preserved ({rate:.1f}%) "
              f"— suspect copies: {s['suspect_copy']} ({copy_rate:.1f}%)")

    if suspect_copy_log:
        print(f"\n🚨 {len(suspect_copy_log)} run(s) with jargon 'preserved' BUT likely "
              f"a pure copy of the English source (no French function word detected) — "
              f"THESE CASES DO NOT COUNT AS A REAL SUCCESS, exclude or requalify them "
              f"before publishing the preservation rate:")
        for rid, quant, translation in suspect_copy_log[:10]:
            print(f"  - {rid} ({quant}): \"{translation[:80]}...\"")
        if len(suspect_copy_log) > 10:
            print(f"  ... and {len(suspect_copy_log) - 10} more, see the full report.")

    if missing_terms_log:
        print(f"\n⚠️ {len(missing_terms_log)} run(s) with at least one missing term — "
              f"REVIEW MANUALLY before publishing a final number "
              f"(possible false negatives, e.g. legitimate rephrasing):")
        for rid, quant, missing, translation in missing_terms_log[:10]:
            print(f"  - {rid} ({quant}): missing={missing} | translation=\"{translation[:80]}...\"")
        if len(missing_terms_log) > 10:
            print(f"  ... and {len(missing_terms_log) - 10} more, see the full report.")


if __name__ == "__main__":
    main()
