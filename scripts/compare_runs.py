#!/usr/bin/env python3
"""Compare two MMMU prediction files on the same question IDs."""
import argparse
from collections import defaultdict
from math import comb
import hashlib
import json
from pathlib import Path


def predictions_path(path):
    return path / "predictions.jsonl" if path.is_dir() else path


def load_rows(path):
    path = predictions_path(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"Duplicate IDs in {path}")
    return by_id


def load_evaluation_config(path):
    """Return the recorded evaluation config and whether it has a signed v1 record."""
    if not path.is_dir():
        return None, False
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return None, False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    signature = manifest.get("evaluation_signature")
    if isinstance(signature, dict) and isinstance(signature.get("config"), dict):
        encoded = json.dumps(signature["config"], ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        if signature.get("sha256") != hashlib.sha256(encoded).hexdigest():
            raise ValueError(f"Invalid evaluation signature in {manifest_path}")
        return signature["config"], True
    arguments = manifest.get("arguments", {})
    # Legacy manifests can still be checked field by field, but lack an explicit pipeline version.
    config = {
        "pipeline_version": manifest.get("evaluation_pipeline_version"),
        "dataset_revision": manifest.get("data_revision"),
        "parser_revision": manifest.get("parser_revision"),
        "image_layout_version": manifest.get("image_layout_version"),
        "sampling_recipe": manifest.get("recipe"),
        "prompt_style": arguments.get("prompt_style", "direct"),
        "mc_prompt": manifest.get("mc_prompt"),
        "open_prompt": manifest.get("open_prompt"),
    }
    for key in ("max_tokens", "max_model_len", "min_pixels", "max_pixels", "batch_size",
                "gpu_memory_utilization"):
        config[key] = arguments.get(key)
    return config, False


def config_differences(config_a, config_b):
    keys = sorted(set(config_a) | set(config_b))
    return [(key, config_a.get(key), config_b.get(key))
            for key in keys if config_a.get(key) != config_b.get(key)]


def exact_mcnemar_p(a_only, b_only):
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, i) for i in range(min(a_only, b_only) + 1)) / (2 ** discordant)
    return min(1.0, 2 * tail)


def metric(rows_a, rows_b, ids):
    n = len(ids)
    correct_a = sum(bool(rows_a[i]["correct"]) for i in ids)
    correct_b = sum(bool(rows_b[i]["correct"]) for i in ids)
    a_only = sum(bool(rows_a[i]["correct"]) and not bool(rows_b[i]["correct"]) for i in ids)
    b_only = sum(bool(rows_b[i]["correct"]) and not bool(rows_a[i]["correct"]) for i in ids)
    return dict(n=n, correct_a=correct_a, correct_b=correct_b,
                accuracy_a=correct_a / n, accuracy_b=correct_b / n,
                delta=(correct_b - correct_a) / n, a_only=a_only, b_only=b_only,
                both_correct=sum(bool(rows_a[i]["correct"]) and bool(rows_b[i]["correct"]) for i in ids),
                both_wrong=sum(not bool(rows_a[i]["correct"]) and not bool(rows_b[i]["correct"]) for i in ids),
                mcnemar_exact_p=exact_mcnemar_p(a_only, b_only))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--output", type=Path, help="Optional Markdown output path")
    parser.add_argument("--allow-config-differences", action="store_true",
                        help="Allow and report intentional evaluation-setting differences for an ablation")
    parser.add_argument("--allow-unverified-config", action="store_true",
                        help="Allow prediction files/runs with no readable manifest (comparison cannot be controlled)")
    args = parser.parse_args()
    config_a, signed_a = load_evaluation_config(args.run_a)
    config_b, signed_b = load_evaluation_config(args.run_b)
    if config_a is None or config_b is None:
        if not args.allow_unverified_config:
            raise SystemExit("Both inputs must be result directories containing manifest.json. "
                             "Use --allow-unverified-config only for legacy prediction files.")
        differences = []
        config_status = "UNVERIFIED: at least one input has no readable manifest."
    else:
        differences = config_differences(config_a, config_b)
        if differences and not args.allow_config_differences:
            preview = "; ".join(f"{key}: {a!r} != {b!r}" for key, a, b in differences[:5])
            raise SystemExit("Evaluation configs differ. For an intentional ablation, rerun with "
                             f"--allow-config-differences. Differences: {preview}")
        if differences:
            config_status = "INTENTIONAL ABLATION: evaluation-setting differences were explicitly allowed."
        elif signed_a and signed_b:
            config_status = "PASS: recorded evaluation signatures match."
        else:
            config_status = "PASS WITH LEGACY MANIFESTS: recorded fields match, but v1 signatures are absent."
    rows_a, rows_b = load_rows(args.run_a), load_rows(args.run_b)
    if set(rows_a) != set(rows_b):
        only_a = sorted(set(rows_a) - set(rows_b))[:5]
        only_b = sorted(set(rows_b) - set(rows_a))[:5]
        raise SystemExit(f"Question ID sets differ: only A={only_a}, only B={only_b}")
    ids = sorted(rows_a)
    groups = {"Overall": ids}
    by_type = defaultdict(list)
    for identifier in ids:
        metadata_a = (rows_a[identifier]["subject"], rows_a[identifier]["question_type"], rows_a[identifier]["answer"])
        metadata_b = (rows_b[identifier]["subject"], rows_b[identifier]["question_type"], rows_b[identifier]["answer"])
        if metadata_a != metadata_b:
            raise SystemExit(f"Question metadata differs for {identifier}")
        by_type[rows_a[identifier]["question_type"]].append(identifier)
    groups.update(by_type)
    results = {name: metric(rows_a, rows_b, group_ids) for name, group_ids in groups.items()}
    lines = [f"# Paired comparison: {args.label_a} vs {args.label_b}", "",
             "## Evaluation configuration", "", config_status]
    if differences:
        lines += ["", "| Setting | A | B |", "|---|---|---|"]
        lines += [f"| `{key}` | `{a}` | `{b}` |" for key, a, b in differences]
    lines += ["", "## Paired accuracy", "",
             "| Group | N | A acc | B acc | B-A | A only | B only | Exact McNemar p |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in results.items():
        lines.append(f"| {name} | {row['n']} | {100*row['accuracy_a']:.2f}% | "
                     f"{100*row['accuracy_b']:.2f}% | {100*row['delta']:+.2f} pp | "
                     f"{row['a_only']} | {row['b_only']} | {row['mcnemar_exact_p']:.4f} |")
    if config_a and config_a.get('benchmark') == 'MMMU-Pro test':
        lines += ["", f"A/B are paired across {len(ids)} MMMU-Pro test questions; the full setting has 1,730 questions."]
    elif len(ids) == 900:
        lines += ["", "A/B are paired across the complete 900-question MMMU validation split."]
    else:
        lines += ["", "A/B are paired by question ID. A development subset is diagnostic only; "
                        "use the full 900-question run for the submitted baseline."]
    output = "\n".join(lines) + "\n"
    print(output, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
