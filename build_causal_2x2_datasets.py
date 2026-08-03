#!/usr/bin/env python3
"""Build the four datasets for the long-tail x contamination control study.

For a fixed dataset-construction seed, the script reuses the repository's
existing injection and pruning manifests to create:

    balanced-clean, balanced-noisy, long-tail-clean, long-tail-noisy

"Balanced" means the original, non-long-tail training distribution. It does
not imply that every semantic class has exactly the same number of images.
All four conditions share the same test and ground-truth splits.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


CONDITIONS: Tuple[Tuple[str, bool, bool], ...] = (
    ("balanced-clean", False, False),
    ("balanced-noisy", False, True),
    ("long-tail-clean", True, False),
    ("long-tail-noisy", True, True),
)

DATASET_MANIFEST_DIRS = {
    "mvtec": "mvtecad-nlt",
    "visa": "visa-nlt",
}

SCENARIOS = ("pareto", "step_k4", "step_k1")


@dataclass(frozen=True)
class BuildInputs:
    dataset: str
    scenario: str
    seed: int
    source_dir: Path
    inject_manifest: Path
    prune_manifest: Path
    inject_entries: Tuple[str, ...]
    prune_entries: Tuple[str, ...]
    injection_sources: Mapping[str, Path]


def _as_manifest_relpath(raw_value: str, manifest_path: Path) -> Tuple[str, Tuple[str, ...]]:
    value = raw_value.replace("\\", "/")
    relpath = PurePosixPath(value)
    parts = relpath.parts
    if relpath.is_absolute() or any(part in ("", ".", "..") for part in parts):
        raise ValueError("unsafe path in {}: {}".format(manifest_path, raw_value))
    if len(parts) != 4 or parts[1:3] != ("train", "good"):
        raise ValueError(
            "manifest entries must have <class>/train/good/<file> form; got {} in {}".format(
                raw_value, manifest_path
            )
        )
    return relpath.as_posix(), parts


def read_manifest(manifest_path: Path) -> Tuple[str, ...]:
    entries: List[str] = []
    seen = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            try:
                normalized, _ = _as_manifest_relpath(value, manifest_path)
            except ValueError as exc:
                raise ValueError("{} (line {})".format(exc, line_number)) from exc
            if normalized in seen:
                raise ValueError(
                    "duplicate path in {} at line {}: {}".format(
                        manifest_path, line_number, normalized
                    )
                )
            seen.add(normalized)
            entries.append(normalized)
    return tuple(entries)


def _dataset_classes(source_dir: Path) -> Tuple[str, ...]:
    class_names = tuple(
        sorted(
            path.name
            for path in source_dir.iterdir()
            if path.is_dir() and (path / "train" / "good").is_dir()
        )
    )
    if not class_names:
        raise ValueError("no <class>/train/good directories found in {}".format(source_dir))
    return class_names


def _resolve_injected_source(source_dir: Path, entry: str, manifest_path: Path) -> Path:
    _, parts = _as_manifest_relpath(entry, manifest_path)
    class_name, target_name = parts[0], parts[3]
    test_dir = source_dir / class_name / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError("test directory required by {} does not exist: {}".format(entry, test_dir))

    matches: List[Path] = []
    for defect_dir in sorted(
        (path for path in test_dir.iterdir() if path.is_dir() and path.name != "good"),
        key=lambda path: (-len(path.name), path.name),
    ):
        prefix = defect_dir.name + "_"
        if not target_name.startswith(prefix):
            continue
        candidate = defect_dir / target_name[len(prefix):]
        if candidate.is_file():
            matches.append(candidate.resolve())

    if not matches:
        raise FileNotFoundError(
            "cannot map injected target {} to a source image under {}".format(entry, test_dir)
        )
    if len(matches) > 1:
        raise ValueError("injected target {} maps to multiple source images: {}".format(entry, matches))
    return matches[0]


def prepare_build_inputs(
    dataset: str,
    scenario: str,
    seed: int,
    source_dir: Path,
    manifest_root: Path,
) -> BuildInputs:
    if dataset not in DATASET_MANIFEST_DIRS:
        raise ValueError("unsupported dataset: {}".format(dataset))
    if scenario not in SCENARIOS:
        raise ValueError("unsupported scenario: {}".format(scenario))
    if seed < 0:
        raise ValueError("seed must be non-negative")

    source_dir = source_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError("source dataset does not exist: {}".format(source_dir))
    class_names = set(_dataset_classes(source_dir))

    seed_name = "seed{:02d}".format(seed)
    manifest_dir = manifest_root.expanduser().resolve() / DATASET_MANIFEST_DIRS[dataset] / scenario / seed_name
    inject_manifest = manifest_dir / "inject_defects.txt"
    prune_manifest = manifest_dir / "prune_good.txt"
    for manifest_path in (inject_manifest, prune_manifest):
        if not manifest_path.is_file():
            raise FileNotFoundError("required manifest does not exist: {}".format(manifest_path))

    inject_entries = read_manifest(inject_manifest)
    prune_entries = read_manifest(prune_manifest)
    injection_sources: Dict[str, Path] = {}

    for entry in prune_entries:
        _, parts = _as_manifest_relpath(entry, prune_manifest)
        if parts[0] not in class_names:
            raise ValueError("unknown class in {}: {}".format(prune_manifest, parts[0]))
        source_path = source_dir.joinpath(*parts)
        if not source_path.is_file():
            raise FileNotFoundError("pruned normal image does not exist: {}".format(source_path))

    for entry in inject_entries:
        _, parts = _as_manifest_relpath(entry, inject_manifest)
        if parts[0] not in class_names:
            raise ValueError("unknown class in {}: {}".format(inject_manifest, parts[0]))
        clean_collision = source_dir.joinpath(*parts)
        if clean_collision.exists() or clean_collision.is_symlink():
            raise ValueError("injected target collides with a clean training image: {}".format(entry))
        injection_sources[entry] = _resolve_injected_source(source_dir, entry, inject_manifest)

    overlap = set(inject_entries).intersection(prune_entries)
    if overlap:
        raise ValueError("injection and pruning manifests overlap: {}".format(sorted(overlap)[:5]))

    return BuildInputs(
        dataset=dataset,
        scenario=scenario,
        seed=seed,
        source_dir=source_dir,
        inject_manifest=inject_manifest.resolve(),
        prune_manifest=prune_manifest.resolve(),
        inject_entries=inject_entries,
        prune_entries=prune_entries,
        injection_sources=injection_sources,
    )


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _materialize_file(source: Path, destination: Path, link_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if link_mode == "symlink":
        os.symlink(str(source.resolve()), str(destination))
    elif link_mode == "hardlink":
        os.link(str(source.resolve()), str(destination))
    elif link_mode == "copy":
        shutil.copy2(str(source), str(destination))
    else:
        raise ValueError("unsupported link mode: {}".format(link_mode))


def _mirror_tree(source: Path, destination: Path, link_mode: str) -> int:
    if not source.is_dir():
        return 0
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for source_file in _iter_files(source):
        relative_path = source_file.relative_to(source)
        _materialize_file(source_file, destination / relative_path, link_mode)
        count += 1
    return count


def _write_manifest(path: Path, entries: Sequence[str]) -> None:
    content = "".join(entry + "\n" for entry in entries)
    path.write_text(content, encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _condition_counts(
    inputs: BuildInputs,
    long_tail: bool,
    noisy: bool,
) -> Dict[str, Dict[str, int]]:
    prune_set = set(inputs.prune_entries) if long_tail else set()
    inject_set = set(inputs.inject_entries) if noisy else set()
    counts: Dict[str, Dict[str, int]] = {}
    for class_name in _dataset_classes(inputs.source_dir):
        source_good = tuple(_iter_files(inputs.source_dir / class_name / "train" / "good"))
        source_relpaths = {
            path.relative_to(inputs.source_dir).as_posix()
            for path in source_good
        }
        num_pruned = len(source_relpaths.intersection(prune_set))
        num_injected = sum(entry.startswith(class_name + "/") for entry in inject_set)
        num_clean = len(source_good) - num_pruned
        counts[class_name] = {
            "source_clean": len(source_good),
            "pruned": num_pruned,
            "retained_clean": num_clean,
            "injected": num_injected,
            "train_total": num_clean + num_injected,
        }
    return counts


def _build_condition(
    inputs: BuildInputs,
    destination: Path,
    condition_name: str,
    long_tail: bool,
    noisy: bool,
    link_mode: str,
    final_condition_path: Path,
) -> Dict[str, object]:
    prune_set = set(inputs.prune_entries) if long_tail else set()
    inject_entries = inputs.inject_entries if noisy else tuple()
    class_names = _dataset_classes(inputs.source_dir)

    for class_name in class_names:
        source_class = inputs.source_dir / class_name
        destination_class = destination / class_name
        destination_good = destination_class / "train" / "good"
        destination_good.mkdir(parents=True, exist_ok=True)

        for source_file in _iter_files(source_class / "train" / "good"):
            source_relative = source_file.relative_to(inputs.source_dir).as_posix()
            if source_relative in prune_set:
                continue
            train_relative = source_file.relative_to(source_class / "train" / "good")
            _materialize_file(source_file, destination_good / train_relative, link_mode)

        _mirror_tree(source_class / "test", destination_class / "test", link_mode)
        _mirror_tree(source_class / "ground_truth", destination_class / "ground_truth", link_mode)

    for entry in inject_entries:
        destination_file = destination.joinpath(*PurePosixPath(entry).parts)
        if destination_file.exists() or destination_file.is_symlink():
            raise FileExistsError("injected destination already exists: {}".format(destination_file))
        _materialize_file(inputs.injection_sources[entry], destination_file, link_mode)

    applied_prune = inputs.prune_entries if long_tail else tuple()
    _write_manifest(destination / "inject_defects.txt", inject_entries)
    _write_manifest(destination / "prune_good.txt", applied_prune)

    per_class = _condition_counts(inputs, long_tail=long_tail, noisy=noisy)
    expected_total = sum(values["train_total"] for values in per_class.values())
    observed_total = sum(
        1
        for class_name in class_names
        for _ in _iter_files(destination / class_name / "train" / "good")
    )
    if observed_total != expected_total:
        raise RuntimeError(
            "{} has {} training files, expected {}".format(
                condition_name, observed_total, expected_total
            )
        )

    metadata = {
        "condition": condition_name,
        "data_path": str(final_condition_path),
        "long_tail": long_tail,
        "noisy": noisy,
        "num_source_clean": sum(values["source_clean"] for values in per_class.values()),
        "num_pruned": sum(values["pruned"] for values in per_class.values()),
        "num_retained_clean": sum(values["retained_clean"] for values in per_class.values()),
        "num_injected": sum(values["injected"] for values in per_class.values()),
        "num_train_total": observed_total,
        "per_class": per_class,
        "diag_manifest_path": str(final_condition_path / "inject_defects.txt"),
    }
    (destination / "build_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def _remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(str(path))


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def build_experiment_group(
    inputs: BuildInputs,
    output_root: Path,
    link_mode: str = "symlink",
    overwrite: bool = False,
) -> Path:
    output_root = output_root.expanduser().resolve()
    if output_root == inputs.source_dir or _is_relative_to(output_root, inputs.source_dir):
        raise ValueError("output root must not be the source dataset or lie inside it")

    seed_name = "seed{:02d}".format(inputs.seed)
    final_group = output_root / inputs.dataset / inputs.scenario / seed_name
    protected_paths = (inputs.source_dir, inputs.inject_manifest, inputs.prune_manifest)
    for protected_path in protected_paths:
        if final_group == protected_path or _is_relative_to(protected_path, final_group):
            raise ValueError(
                "output seed group must not contain a source or manifest path: {}".format(
                    protected_path
                )
            )
    final_group.parent.mkdir(parents=True, exist_ok=True)
    if final_group.exists() or final_group.is_symlink():
        if not overwrite:
            raise FileExistsError(
                "output already exists: {} (pass --overwrite to replace this seed group)".format(
                    final_group
                )
            )

    temporary_group = Path(
        tempfile.mkdtemp(prefix=".{}-".format(seed_name), dir=str(final_group.parent))
    )
    try:
        condition_summaries: Dict[str, object] = {}
        for condition_name, long_tail, noisy in CONDITIONS:
            condition_summaries[condition_name] = _build_condition(
                inputs=inputs,
                destination=temporary_group / condition_name,
                condition_name=condition_name,
                long_tail=long_tail,
                noisy=noisy,
                link_mode=link_mode,
                final_condition_path=final_group / condition_name,
            )

        summary = {
            "schema_version": 1,
            "dataset": inputs.dataset,
            "scenario": inputs.scenario,
            "seed": inputs.seed,
            "seed_name": seed_name,
            "source_dir": str(inputs.source_dir),
            "link_mode": link_mode,
            "balanced_definition": "original non-long-tail training distribution",
            "test_protocol": "all four conditions share the complete original test split",
            "injected_images_also_in_test": len(inputs.inject_entries),
            "input_manifests": {
                "inject_defects": str(inputs.inject_manifest),
                "inject_defects_sha256": _sha256(inputs.inject_manifest),
                "prune_good": str(inputs.prune_manifest),
                "prune_good_sha256": _sha256(inputs.prune_manifest),
            },
            "conditions": condition_summaries,
        }
        (temporary_group / "build_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_manifest(temporary_group / "inject_defects.txt", inputs.inject_entries)
        _write_manifest(temporary_group / "prune_good.txt", inputs.prune_entries)

        if final_group.exists() or final_group.is_symlink():
            _remove_existing(final_group)
        os.replace(str(temporary_group), str(final_group))
    except Exception:
        if temporary_group.exists():
            shutil.rmtree(str(temporary_group))
        raise

    return final_group


def _normalize_choices(values: Sequence[str], all_values: Sequence[str], option_name: str) -> Tuple[str, ...]:
    if "all" in values:
        if len(values) != 1:
            raise ValueError("{}: 'all' cannot be combined with other values".format(option_name))
        return tuple(all_values)
    return tuple(dict.fromkeys(values))


def parse_args() -> argparse.Namespace:
    repo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build balanced/long-tail x clean/noisy datasets from existing manifests."
    )
    parser.add_argument("--seed", type=int, required=True, help="Dataset-construction seed, e.g. 1 for seed01")
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=("mvtec", "visa", "all"),
        default=("all",),
        help="Datasets to build (default: all)",
    )
    parser.add_argument(
        "--scenario",
        nargs="+",
        choices=SCENARIOS + ("all",),
        default=("step_k1",),
        help="Long-tail scenarios to build (default: step_k1)",
    )
    parser.add_argument(
        "--mvtec-source",
        type=Path,
        default=repo_dir.parent / "mvtec_anomaly_detection",
        help="Pristine MVTec AD root",
    )
    parser.add_argument(
        "--visa-source",
        type=Path,
        default=repo_dir.parent / "visa_",
        help="MVTec-format VisA root",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=repo_dir / "manifest",
        help="Root containing mvtecad-nlt and visa-nlt manifests",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_dir.parent / "causal_2x2_datasets",
        help="Output root",
    )
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How source images are materialized (default: symlink)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing dataset/scenario/seed output group",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print planned output paths without writing data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        datasets = _normalize_choices(args.dataset, tuple(DATASET_MANIFEST_DIRS), "--dataset")
        scenarios = _normalize_choices(args.scenario, SCENARIOS, "--scenario")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    source_dirs = {
        "mvtec": args.mvtec_source,
        "visa": args.visa_source,
    }
    all_inputs: List[BuildInputs] = []
    for dataset in datasets:
        for scenario in scenarios:
            all_inputs.append(
                prepare_build_inputs(
                    dataset=dataset,
                    scenario=scenario,
                    seed=args.seed,
                    source_dir=source_dirs[dataset],
                    manifest_root=args.manifest_root,
                )
            )

    output_root = args.output_root.expanduser().resolve()
    for inputs in all_inputs:
        planned_group = output_root / inputs.dataset / inputs.scenario / "seed{:02d}".format(inputs.seed)
        print(
            "[plan] dataset={} scenario={} seed={:02d} inject={} prune={} output={}".format(
                inputs.dataset,
                inputs.scenario,
                inputs.seed,
                len(inputs.inject_entries),
                len(inputs.prune_entries),
                planned_group,
            )
        )

    if args.dry_run:
        print("[done] dry run validated all sources and manifests; no files were written")
        return

    for inputs in all_inputs:
        output_group = build_experiment_group(
            inputs=inputs,
            output_root=output_root,
            link_mode=args.link_mode,
            overwrite=args.overwrite,
        )
        print("[done] built four conditions under {}".format(output_group))
    print("[done] causal 2x2 dataset construction completed")


if __name__ == "__main__":
    main()
