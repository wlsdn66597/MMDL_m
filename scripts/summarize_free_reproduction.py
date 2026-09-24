#!/usr/bin/env python3
"""Recheck coverage/scoring of saved free32768 runs, without GPU inference."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_mmmu as val
from baseline_contract import RUNS
from eval_free_reproduction import validate_run


def summarize_suite(root):
    root = Path(root)
    results, table, shared, pro_ids = {}, [], None, None
    columns = ["run", "n", "accuracy_pct", "macro_accuracy_pct", "unparsed", "length_limited",
               "mean_output_tokens", "inference_minutes", "total_minutes", "calls"]
    for name, benchmark, setting in RUNS:
        manifest, summary, rows = validate_run(root / name)
        identity = manifest["identity"]
        if (identity["benchmark"], identity["setting"]) != (benchmark, setting):
            raise ValueError(f"Wrong dataset/setting in {name}")
        current = {k: identity[k] for k in ("model_path", "model_revision", "profile", "code_sha256")}
        if shared is not None and current != shared:
            raise ValueError("Suite mixes model checkpoints or evaluation configurations")
        shared = current
        if benchmark == "mmmu-pro":
            ids = {r["id"] for r in rows}
            if pro_ids is not None and ids != pro_ids:
                raise ValueError("MMMU-Pro settings do not cover the same IDs")
            pro_ids = ids
        results[name] = summary
        table.append([name, str(summary["n"]), f"{summary['accuracy']*100:.2f}",
                      f"{summary['macro_accuracy']*100:.2f}", str(summary["unparsed"]),
                      str(summary["length_limited"]), f"{summary['output_tokens']/summary['n']:.1f}",
                      f"{summary['inference_seconds']/60:.1f}", f"{summary['total_seconds']/60:.1f}",
                      str(summary["calls"])])
    tsv = "\n".join("\t".join(row) for row in [columns, *table]) + "\n"
    (root / "summary.tsv").write_text(tsv, encoding="utf-8")
    lines = ["# Free generation with Qwen reference prompts (32768-token ceiling)", "",
             "Coverage: MMMU validation 900 (847 MC + 53 open); MMMU-Pro 1730 per setting.",
             "One unconstrained response per question. No second answer-generation call.", "",
             "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(row) + " |" for row in table]
    delta = results["mmmu_val"]["macro_accuracy"] * 100 - 67.4
    lines += ["", f"MMMU reference: Qwen report 67.4%; this deterministic score differs by {delta:+.2f} pp.",
              "This is not an exact reproduction of the Qwen judge-based score. See docs/free32768_reproduction.md.",
              "Accuracy uses all questions, including unparsed/truncated responses. No random answer fallback.",
              "total_minutes sums execution sessions (including failed attempts); inference_minutes sums saved completions.",
              "The earlier 8192 free and two-stage runs have different prompts/policies; this is not a token-only ablation."]
    (root / "baseline_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    val.write_json(root / "baseline_summary.json", {"identity": shared, "runs": results})
    print(tsv, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    summarize_suite(parser.parse_args().root)
