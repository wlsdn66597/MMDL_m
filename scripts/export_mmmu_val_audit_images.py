#!/usr/bin/env python3
"""Export MMMU-val images that match a saved two-stage audit's image hashes.

The archive contains image PNGs and a mapping by question ID. It intentionally
omits predictions, labels, and prompts because those are already in the audit.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline_contract import validate_run
from eval_mmmu import DATA_REV, SUBJECTS, image_as_rgb


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def add_bytes(archive, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True,
                        help="Completed MMMU-val two-stage run with predictions.jsonl and inputs.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="New .tar.gz file")
    parser.add_argument("--scope", choices=("all", "wrong"), default="wrong")
    parser.add_argument("--ids-file", type=Path,
                        help="Optional newline-separated IDs, intersected with --scope")
    parser.add_argument("--data-root", default="MMMU/MMMU",
                        help="Official dataset or pinned local snapshot directory")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--plan-only", action="store_true",
                        help="Verify the saved audit and count images without loading the dataset")
    args = parser.parse_args()

    manifest, _, predictions = validate_run(args.audit_dir, "mmmu-val", "standard")
    inputs_list = read_jsonl(args.audit_dir / "inputs.jsonl")
    inputs = {row["id"]: row for row in inputs_list}
    if len(inputs) != len(inputs_list) or set(inputs) != {row["id"] for row in predictions}:
        raise ValueError("inputs.jsonl must cover exactly the saved prediction IDs")

    selected = {row["id"] for row in predictions if args.scope == "all" or not row["correct"]}
    if args.ids_file:
        requested = [line.strip() for line in args.ids_file.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
        if len(requested) != len(set(requested)) or set(requested) - set(inputs):
            raise ValueError("ids-file has duplicate or unknown IDs")
        selected &= set(requested)
    if not selected:
        raise ValueError("No IDs selected")
    expected_images = sum(len(inputs[item_id]["images"]) for item_id in selected)
    print(f"[plan] scope={args.scope} questions={len(selected)} images={expected_images}", flush=True)
    if args.plan_only:
        return

    data_root = Path(args.data_root).expanduser()
    if args.data_root == "MMMU/MMMU":
        source = args.data_root
        load_kwargs = {"revision": DATA_REV}
    elif data_root.is_dir() and data_root.name == DATA_REV:
        source = str(data_root.resolve())
        load_kwargs = {}
    else:
        raise ValueError(f"Use MMMU/MMMU or its pinned {DATA_REV} snapshot directory")
    if args.cache_dir:
        load_kwargs["cache_dir"] = args.cache_dir
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.output.suffixes[-2:] != [".tar", ".gz"]:
        raise ValueError("--output must end with .tar.gz")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".mmmu-images-", suffix=".tar.gz",
                                         dir=args.output.parent, delete=False) as tmp:
            temporary = Path(tmp.name)
        found_ids = set()
        exported = []
        with tarfile.open(temporary, "w:gz") as archive:
            for subject in SUBJECTS:
                expected_subject_ids = {row["id"] for row in predictions if row["subject"] == subject}
                if not expected_subject_ids & selected:
                    continue
                dataset = load_dataset(source, subject, split="validation", **load_kwargs)
                if dataset._fingerprint != manifest["dataset_fingerprints"][subject]:
                    raise ValueError(f"Dataset fingerprint differs for {subject}")
                if set(dataset["id"]) != expected_subject_ids:
                    raise ValueError(f"Dataset IDs differ for {subject}")
                for example in dataset:
                    item_id = example["id"]
                    if item_id not in selected:
                        continue
                    if not re.fullmatch(r"[A-Za-z0-9_-]+", item_id):
                        raise ValueError(f"Unsafe question ID: {item_id!r}")
                    found_ids.add(item_id)
                    expected = {item["number"]: item for item in inputs[item_id]["images"]}
                    present = {i for i in range(1, 8) if example.get(f"image_{i}") is not None}
                    if present != set(expected):
                        raise ValueError(f"Image numbers differ for {item_id}")
                    for number in sorted(expected):
                        original = example[f"image_{number}"]
                        buffer = io.BytesIO()
                        image_as_rgb(original).save(buffer, format="PNG")
                        png = buffer.getvalue()
                        sha = hashlib.sha256(png).hexdigest()
                        meta = expected[number]
                        if (sha != meta["png_sha256"] or original.width != meta["width"]
                                or original.height != meta["height"]):
                            raise ValueError(f"Image/hash mismatch for {item_id} image {number}")
                        path = f"images/{item_id}/image_{number}.png"
                        add_bytes(archive, path, png)
                        exported.append({"id": item_id, "subject": subject, "number": number,
                                         "path": path, "png_sha256": sha,
                                         "width": original.width, "height": original.height})
                print(f"[data] {subject}: {len(found_ids & selected)} selected IDs exported so far", flush=True)
            if found_ids != selected or len(exported) != expected_images:
                raise ValueError(f"Incomplete export: IDs {len(found_ids)}/{len(selected)}, "
                                 f"images {len(exported)}/{expected_images}")
            listing = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in exported)
            add_bytes(archive, "images_manifest.jsonl", listing.encode("utf-8"))
            summary = {"dataset": "MMMU/MMMU", "revision": DATA_REV, "scope": args.scope,
                       "questions": len(found_ids), "images": len(exported),
                       "audit_profile_sha256": manifest["evaluation_profile"]["sha256"]}
            add_bytes(archive, "export_summary.json",
                      (json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        os.rename(temporary, args.output)
        print(f"[done] {args.output}: {len(found_ids)} questions, {len(exported)} images", flush=True)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    main()
