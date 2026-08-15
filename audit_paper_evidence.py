#!/usr/bin/env python3
"""Recompute paper-facing evidence from immutable experiment artifacts.

    The script never writes into the result tree. It reads the six seed01 v5 H-only
runs and the raw causal 2x2 logs, then writes auditable CSV/JSON summaries to
``analysis_outputs/paper_evidence_v1`` by default. Only Python's standard
library is required so the audit does not depend on the training environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


SCENARIOS: Tuple[Tuple[str, str], ...] = (
    ("mvtec", "pareto"),
    ("mvtec", "step_k4"),
    ("mvtec", "step_k1"),
    ("visa", "pareto"),
    ("visa", "step_k4"),
    ("visa", "step_k1"),
)

CONDITIONS: Tuple[str, ...] = (
    "balanced_clean",
    "balanced_noisy",
    "long_tail_clean",
    "long_tail_noisy",
)

CONDITION_CODE = {
    "balanced_clean": "Y00",
    "balanced_noisy": "Y01",
    "long_tail_clean": "Y10",
    "long_tail_noisy": "Y11",
}

METRICS: Tuple[str, ...] = (
    "I-AUROC",
    "I-AP",
    "I-F1",
    "P-AUROC",
    "P-AP",
    "P-F1",
    "P-AUPRO",
)

LOG_METRIC_RE = re.compile(
    r"^(?P<label>[^:]+):\s*"
    r"I-Auroc:(?P<I_AUROC>[-+0-9.eE]+),\s*"
    r"I-AP:(?P<I_AP>[-+0-9.eE]+),\s*"
    r"I-F1:(?P<I_F1>[-+0-9.eE]+),\s*"
    r"P-AUROC:(?P<P_AUROC>[-+0-9.eE]+),\s*"
    r"P-AP:(?P<P_AP>[-+0-9.eE]+),\s*"
    r"P-F1:(?P<P_F1>[-+0-9.eE]+),\s*"
    r"P-AUPRO:(?P<P_AUPRO>[-+0-9.eE]+)\s*$"
)


def scenario_name(dataset: str, setting: str) -> str:
    return "{}_{}_seed01".format(dataset, setting)


def require_file(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError("Required audit input is missing: {}".format(path))
    return path


def read_csv(path: Path) -> List[Dict[str, str]]:
    path = require_file(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Dict[str, Any]:
    path = require_file(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object: {}".format(path))
    return payload


def as_int(value: Any) -> int:
    return int(float(value))


def ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError("Metric denominator must be positive")
    return float(numerator) / float(denominator)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("Cannot average an empty collection")
    return sum(values) / len(values)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SourceRegistry:
    def __init__(self) -> None:
        self._records: MutableMapping[str, Dict[str, Any]] = {}

    def add(self, path: Path, role: str, scenario: str = "") -> Path:
        resolved = require_file(path)
        key = str(resolved)
        record = self._records.get(key)
        if record is None:
            stat = resolved.stat()
            record = {
                "path": key,
                "size_bytes": stat.st_size,
                "sha256": sha256(resolved),
                "roles": set(),
                "scenarios": set(),
            }
            self._records[key] = record
        record["roles"].add(role)
        if scenario:
            record["scenarios"].add(scenario)
        return resolved

    def rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for path in sorted(self._records):
            record = self._records[path]
            rows.append(
                {
                    "path": record["path"],
                    "size_bytes": record["size_bytes"],
                    "sha256": record["sha256"],
                    "roles": ";".join(sorted(record["roles"])),
                    "scenarios": ";".join(sorted(record["scenarios"])),
                }
            )
        return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def binary_rank_metrics(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, Any]:
    """Match sklearn's AUROC and non-interpolated average precision."""
    if len(labels) != len(scores) or not labels:
        raise ValueError("Binary metric inputs must be non-empty and aligned")
    items = sorted((float(score), int(label)) for label, score in zip(labels, scores))
    positives = sum(label for _, label in items)
    negatives = len(items) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Both binary classes are required")

    wins = 0.0
    negatives_before = 0
    index = 0
    while index < len(items):
        end = index + 1
        while end < len(items) and items[end][0] == items[index][0]:
            end += 1
        group = items[index:end]
        group_positive = sum(label for _, label in group)
        group_negative = len(group) - group_positive
        wins += group_positive * negatives_before + 0.5 * group_positive * group_negative
        negatives_before += group_negative
        index = end
    auroc = wins / (positives * negatives)

    true_positive = 0
    predicted_positive = 0
    previous_recall = 0.0
    average_precision = 0.0
    descending = sorted(items, reverse=True)
    index = 0
    while index < len(descending):
        end = index + 1
        while end < len(descending) and descending[end][0] == descending[index][0]:
            end += 1
        group = descending[index:end]
        true_positive += sum(label for _, label in group)
        predicted_positive += len(group)
        recall = true_positive / positives
        precision = true_positive / predicted_positive
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        index = end

    return {
        "auroc": auroc,
        "average_precision": average_precision,
        "num_samples": len(items),
        "num_positive": positives,
        "num_negative": negatives,
    }


def cleanup_metrics(
    sample_ids: set,
    contaminated_ids: set,
    clean_tail_ids: set,
    removed_ids: set,
) -> Dict[str, Any]:
    if not removed_ids <= sample_ids:
        raise ValueError("Removal set contains unknown samples")
    clean_ids = sample_ids - contaminated_ids
    retained_ids = sample_ids - removed_ids
    noise_removed = removed_ids & contaminated_ids
    clean_removed = removed_ids & clean_ids
    clean_tail_removed = removed_ids & clean_tail_ids
    return {
        "num_initial": len(sample_ids),
        "num_removed": len(removed_ids),
        "num_contaminated_initial": len(contaminated_ids),
        "num_noise_removed": len(noise_removed),
        "num_clean_initial": len(clean_ids),
        "num_clean_removed": len(clean_removed),
        "num_clean_gt_tail_initial": len(clean_tail_ids),
        "num_clean_gt_tail_removed": len(clean_tail_removed),
        "removed_set_noise_precision": ratio(len(noise_removed), len(removed_ids)),
        "noise_removal_recall": ratio(len(noise_removed), len(contaminated_ids)),
        "clean_retention_rate": ratio(len(retained_ids & clean_ids), len(clean_ids)),
        "clean_gt_tail_removal_rate": ratio(len(clean_tail_removed), len(clean_tail_ids)),
        "residual_contamination_rate": ratio(
            len(retained_ids & contaminated_ids), len(retained_ids)
        ),
    }


def audit_v5(
    results_root: Path, sources: SourceRegistry
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    v5_root = results_root / "dependency_ablation_v5"
    purification_rows: List[Dict[str, Any]] = []
    warmup_rows: List[Dict[str, Any]] = []
    tail_rows: List[Dict[str, Any]] = []

    for dataset, setting in SCENARIOS:
        scenario = scenario_name(dataset, setting)
        run_dir = v5_root / "{}_h_only".format(scenario)
        tailguard_dir = run_dir / "tailguard"

        summary_path = sources.add(
            tailguard_dir / "tailguard_summary.json", "v5_selected_checkpoint", scenario
        )
        details_path = sources.add(
            tailguard_dir
            / "prepare"
            / "tail_sampler_analysis_only"
            / "sampler_analysis_details.csv",
            "v5_analysis_labels_and_tail_candidates",
            scenario,
        )
        removed_path = sources.add(
            tailguard_dir / "stage2" / "stage2_removed_samples.csv",
            "v5_h_removal_set",
            scenario,
        )
        summary = read_json(summary_path)
        selected_iteration = as_int(
            summary["gbps_trigger_summary"]["gbps_selected_iter"]
        )
        selected_dir = tailguard_dir / "gbps" / "iter_{:05d}".format(selected_iteration)
        train_scores_path = sources.add(
            selected_dir / "train_scores.csv", "v5_selected_global_residual", scenario
        )
        group_scores_path = sources.add(
            selected_dir / "h_sample_group_scores.csv",
            "v5_selected_supported_group_scores",
            scenario,
        )

        details = read_csv(details_path)
        detail_by_id = {as_int(row["sample_idx"]): row for row in details}
        if len(detail_by_id) != len(details):
            raise ValueError("Duplicate sample_idx in {}".format(details_path))
        sample_ids = set(detail_by_id)
        contaminated_ids = {
            sample_idx
            for sample_idx, row in detail_by_id.items()
            if as_int(row["is_contaminated"]) == 1
        }
        clean_tail_ids = {
            sample_idx
            for sample_idx, row in detail_by_id.items()
            if as_int(row["is_contaminated"]) == 0 and as_int(row["is_gt_tail"]) == 1
        }

        h_removed_ids = {as_int(row["sample_idx"]) for row in read_csv(removed_path)}
        train_scores = read_csv(train_scores_path)
        score_by_id = {
            as_int(row["sample_idx"]): float(row["image_score"]) for row in train_scores
        }
        if set(score_by_id) != sample_ids:
            raise ValueError("Selected train scores do not cover the full training set: {}".format(scenario))
        matched_count = len(h_removed_ids)
        global_removed_ids = {
            sample_idx
            for sample_idx, _ in sorted(
                score_by_id.items(), key=lambda item: (-item[1], item[0])
            )[:matched_count]
        }
        for strategy, removed_ids, removal_source in (
            ("global_matched_count", global_removed_ids, str(train_scores_path)),
            ("H_group_conditional", h_removed_ids, str(removed_path)),
        ):
            metrics = cleanup_metrics(
                sample_ids, contaminated_ids, clean_tail_ids, removed_ids
            )
            purification_rows.append(
                {
                    "dataset": dataset,
                    "setting": setting,
                    "scenario": scenario,
                    "strategy": strategy,
                    "selected_iteration": selected_iteration,
                    "removal_source": removal_source,
                    **metrics,
                }
            )

        group_scores = read_csv(group_scores_path)
        group_ids = [as_int(row["sample_idx"]) for row in group_scores]
        if len(set(group_ids)) != len(group_ids) or not set(group_ids) <= sample_ids:
            raise ValueError("Invalid supported-group score sample IDs: {}".format(scenario))
        if any(as_int(detail_by_id[sample_idx]["tail_candidate"]) != 0 for sample_idx in group_ids):
            raise ValueError("Warm-up scores include a tail candidate: {}".format(scenario))
        labels = [as_int(detail_by_id[sample_idx]["is_contaminated"]) for sample_idx in group_ids]
        for signal in ("image_score", "z_score", "p_high"):
            metrics = binary_rank_metrics(labels, [float(row[signal]) for row in group_scores])
            warmup_rows.append(
                {
                    "dataset": dataset,
                    "setting": setting,
                    "scenario": scenario,
                    "selected_iteration": selected_iteration,
                    "population": "supported_candidates_H0",
                    "signal": signal,
                    "score_source": str(group_scores_path),
                    **metrics,
                }
            )

        tail_candidate_ids = {
            sample_idx
            for sample_idx, row in detail_by_id.items()
            if as_int(row["tail_candidate"]) == 1
        }
        clean_tail_count = sum(
            as_int(detail_by_id[sample_idx]["is_contaminated"]) == 0
            and as_int(detail_by_id[sample_idx]["is_gt_tail"]) == 1
            for sample_idx in tail_candidate_ids
        )
        clean_head_count = sum(
            as_int(detail_by_id[sample_idx]["is_contaminated"]) == 0
            and as_int(detail_by_id[sample_idx]["is_gt_tail"]) == 0
            for sample_idx in tail_candidate_ids
        )
        contamination_count = sum(
            as_int(detail_by_id[sample_idx]["is_contaminated"]) == 1
            for sample_idx in tail_candidate_ids
        )
        tail_rows.append(
            {
                "dataset": dataset,
                "setting": setting,
                "scenario": scenario,
                "num_T0": len(tail_candidate_ids),
                "num_T0_removed_by_H": len(tail_candidate_ids & h_removed_ids),
                "T0_clean_gt_tail": clean_tail_count,
                "T0_clean_gt_head": clean_head_count,
                "T0_contamination": contamination_count,
                "candidate_source": str(details_path),
                "h_removal_source": str(removed_path),
            }
        )

    purification_macro: List[Dict[str, Any]] = []
    cleanup_rate_fields = (
        "removed_set_noise_precision",
        "noise_removal_recall",
        "clean_retention_rate",
        "clean_gt_tail_removal_rate",
        "residual_contamination_rate",
    )
    for strategy in ("global_matched_count", "H_group_conditional"):
        selected = [row for row in purification_rows if row["strategy"] == strategy]
        row: Dict[str, Any] = {
            "strategy": strategy,
            "num_scenarios": len(selected),
            "macro_averaging": "unweighted_arithmetic_mean_over_six_scenarios",
            "sum_num_initial": sum(row["num_initial"] for row in selected),
            "sum_num_removed": sum(row["num_removed"] for row in selected),
        }
        for field in cleanup_rate_fields:
            row[field] = mean(item[field] for item in selected)
            row[field + "_percent"] = 100.0 * row[field]
        purification_macro.append(row)

    warmup_macro: List[Dict[str, Any]] = []
    for signal in ("image_score", "z_score", "p_high"):
        selected = [row for row in warmup_rows if row["signal"] == signal]
        warmup_macro.append(
            {
                "signal": signal,
                "population": "supported_candidates_H0",
                "num_scenarios": len(selected),
                "macro_averaging": "unweighted_arithmetic_mean_over_six_scenarios",
                "auroc": mean(row["auroc"] for row in selected),
                "auroc_percent": 100.0 * mean(row["auroc"] for row in selected),
                "average_precision": mean(row["average_precision"] for row in selected),
                "average_precision_percent": 100.0
                * mean(row["average_precision"] for row in selected),
                "sum_num_samples": sum(row["num_samples"] for row in selected),
                "sum_num_positive": sum(row["num_positive"] for row in selected),
                "sum_num_negative": sum(row["num_negative"] for row in selected),
            }
        )

    tail_total_fields = (
        "num_T0",
        "num_T0_removed_by_H",
        "T0_clean_gt_tail",
        "T0_clean_gt_head",
        "T0_contamination",
    )
    tail_summary = {
        "num_scenarios": len(tail_rows),
        **{field: sum(row[field] for row in tail_rows) for field in tail_total_fields},
    }
    if tail_summary["num_T0"] != (
        tail_summary["T0_clean_gt_tail"]
        + tail_summary["T0_clean_gt_head"]
        + tail_summary["T0_contamination"]
    ):
        raise AssertionError("Aggregate T0 composition does not balance")

    v5_summary = {
        "purification": {row["strategy"]: row for row in purification_macro},
        "warmup": {row["signal"]: row for row in warmup_macro},
        "tail_flow": tail_summary,
    }
    return (
        purification_rows,
        purification_macro,
        warmup_rows,
        warmup_macro,
        tail_rows,
        v5_summary,
    )


def parse_causal_log(path: Path) -> Dict[str, Any]:
    text = require_file(path).read_text(encoding="utf-8", errors="replace")
    summary: Dict[str, float] = {}
    per_class: Dict[str, Dict[str, float]] = {}
    for line in text.splitlines():
        match = LOG_METRIC_RE.match(line.strip())
        if not match:
            continue
        values = {
            "I-AUROC": float(match.group("I_AUROC")),
            "I-AP": float(match.group("I_AP")),
            "I-F1": float(match.group("I_F1")),
            "P-AUROC": float(match.group("P_AUROC")),
            "P-AP": float(match.group("P_AP")),
            "P-F1": float(match.group("P_F1")),
            "P-AUPRO": float(match.group("P_AUPRO")),
        }
        if match.group("label") == "Mean":
            summary = values
        else:
            # Logs contain intermediate evaluations; the last occurrence is final.
            per_class[match.group("label")] = values
    if set(summary) != set(METRICS):
        raise ValueError("No complete final Mean row in {}".format(path))
    train_matches = re.findall(r"train image number:\s*(\d+)", text)
    if not train_matches:
        raise ValueError("No train image count in {}".format(path))
    return {
        "summary": summary,
        "per_class": per_class,
        "train_images": int(train_matches[-1]),
        "completed_10000": bool(
            "Training finished." in text
            and (
                re.search(r"iter \[(?:9999|10000)/10000\]", text)
                or re.search(r"total_iters\s+10000", text)
            )
        ),
    }


def causal_path(
    results_root: Path,
    dataset: str,
    setting: str,
    condition: str,
    balanced_clean_logs: Mapping[str, Path],
) -> Path:
    scenario = scenario_name(dataset, setting)
    if condition == "balanced_clean":
        return balanced_clean_logs[dataset]
    if condition in ("balanced_noisy", "long_tail_clean"):
        return (
            results_root
            / "causal_2x2_original"
            / "{}_{}_original_dinomaly".format(scenario, condition)
            / "log.txt"
        )
    if condition == "long_tail_noisy":
        return (
            results_root
            / "original_dinomaly_baseline"
            / "{}_original_dinomaly".format(scenario)
            / "log.txt"
        )
    raise ValueError("Unknown causal condition: {}".format(condition))


def validate_causal_csvs(
    causal_analysis_dir: Path,
    cells: Sequence[Mapping[str, Any]],
    class_cells: Sequence[Mapping[str, Any]],
    class_effects: Sequence[Mapping[str, Any]],
    sources: SourceRegistry,
) -> Dict[str, Any]:
    paths = {
        "cells_summary": causal_analysis_dir / "cells_summary.csv",
        "cells_per_class": causal_analysis_dir / "cells_per_class.csv",
        "effects_per_class": causal_analysis_dir / "effects_per_class.csv",
    }
    if not all(path.is_file() for path in paths.values()):
        return {
            "status": "skipped_missing_reference_csv",
            "paths": {key: str(path.resolve()) for key, path in paths.items()},
        }
    for role, path in paths.items():
        sources.add(path, "causal_reference_{}_csv".format(role))

    comparison_metrics = ("I-AUROC", "I-AP", "P-AUROC", "P-AP", "P-AUPRO")
    reference_cells = {
        (row["dataset"], row["scenario"], row["condition"]): row
        for row in read_csv(paths["cells_summary"])
    }
    cell_mismatches = 0
    for row in cells:
        key = (row["dataset"], row["setting"], row["condition_label"])
        reference = reference_cells.get(key)
        if reference is None:
            cell_mismatches += 1
            continue
        if as_int(reference["train_images"]) != row["train_images"]:
            cell_mismatches += 1
            continue
        if any(float(reference[metric]) != row[metric] for metric in comparison_metrics):
            cell_mismatches += 1

    reference_class_cells = {
        (row["dataset"], row["scenario"], row["condition"], row["class"]): row
        for row in read_csv(paths["cells_per_class"])
    }
    class_cell_mismatches = 0
    for row in class_cells:
        key = (
            row["dataset"],
            row["setting"],
            row["condition_label"],
            row["class_name"],
        )
        reference = reference_class_cells.get(key)
        if reference is None or any(
            float(reference[metric]) != row[metric] for metric in comparison_metrics
        ):
            class_cell_mismatches += 1

    reference_effects = {
        (row["dataset"], row["scenario"], row["class"], row["metric"]): row
        for row in read_csv(paths["effects_per_class"])
    }
    comparable_class_effects = [
        row for row in class_effects if row["metric"] in comparison_metrics
    ]
    class_effect_mismatches = 0
    for row in comparable_class_effects:
        key = (row["dataset"], row["setting"], row["class_name"], row["metric"])
        reference = reference_effects.get(key)
        if reference is None or abs(float(reference["interaction"]) - row["interaction"]) > 1e-12:
            class_effect_mismatches += 1

    return {
        "status": "matched" if not any(
            (cell_mismatches, class_cell_mismatches, class_effect_mismatches)
        ) else "mismatch",
        "recomputed_cell_rows": len(cells),
        "reference_cell_rows": len(reference_cells),
        "cell_mismatches": cell_mismatches,
        "recomputed_class_cell_rows": len(class_cells),
        "reference_class_cell_rows": len(reference_class_cells),
        "class_cell_mismatches": class_cell_mismatches,
        "recomputed_class_effect_rows": len(class_effects),
        "reference_class_effect_rows": len(reference_effects),
        "compared_class_effect_rows": len(comparable_class_effects),
        "class_effect_mismatches": class_effect_mismatches,
        "comparison_metrics": list(comparison_metrics),
        "additional_recomputed_metrics": sorted(set(METRICS).difference(comparison_metrics)),
    }


def audit_causal(
    results_root: Path,
    balanced_clean_logs: Mapping[str, Path],
    causal_analysis_dir: Path,
    sources: SourceRegistry,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    cells: List[Dict[str, Any]] = []
    class_cells: List[Dict[str, Any]] = []
    parsed: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for dataset, setting in SCENARIOS:
        scenario = scenario_name(dataset, setting)
        for condition in CONDITIONS:
            path = causal_path(
                results_root, dataset, setting, condition, balanced_clean_logs
            )
            path = sources.add(path, "causal_{}_log".format(condition), scenario)
            payload = parse_causal_log(path)
            expected_classes = 15 if dataset == "mvtec" else 12
            if len(payload["per_class"]) != expected_classes:
                raise ValueError(
                    "Expected {} final classes in {}, found {}".format(
                        expected_classes, path, len(payload["per_class"])
                    )
                )
            if not payload["completed_10000"]:
                raise ValueError("Causal run is not complete at 10000 iterations: {}".format(path))
            parsed[(dataset, setting, condition)] = payload
            cells.append(
                {
                    "dataset": dataset,
                    "setting": setting,
                    "scenario": scenario,
                    "condition": condition,
                    "condition_label": "{}_{}".format(CONDITION_CODE[condition], condition),
                    "long_tail": int(condition.startswith("long_tail")),
                    "noisy": int(condition.endswith("noisy")),
                    "train_images": payload["train_images"],
                    "completed_10000": payload["completed_10000"],
                    "source_log": str(path),
                    **payload["summary"],
                }
            )
            for class_name, metrics in sorted(payload["per_class"].items()):
                class_cells.append(
                    {
                        "dataset": dataset,
                        "setting": setting,
                        "scenario": scenario,
                        "condition": condition,
                        "condition_label": "{}_{}".format(CONDITION_CODE[condition], condition),
                        "class_name": class_name,
                        "source_log": str(path),
                        **metrics,
                    }
                )

    scenario_effects: List[Dict[str, Any]] = []
    class_effects: List[Dict[str, Any]] = []
    for dataset, setting in SCENARIOS:
        scenario = scenario_name(dataset, setting)
        condition_payloads = {
            condition: parsed[(dataset, setting, condition)] for condition in CONDITIONS
        }
        class_sets = [set(payload["per_class"]) for payload in condition_payloads.values()]
        if any(class_set != class_sets[0] for class_set in class_sets[1:]):
            raise ValueError("Causal class sets differ across conditions: {}".format(scenario))
        for metric in METRICS:
            y00 = condition_payloads["balanced_clean"]["summary"][metric]
            y01 = condition_payloads["balanced_noisy"]["summary"][metric]
            y10 = condition_payloads["long_tail_clean"]["summary"][metric]
            y11 = condition_payloads["long_tail_noisy"]["summary"][metric]
            scenario_effects.append(
                {
                    "dataset": dataset,
                    "setting": setting,
                    "scenario": scenario,
                    "metric": metric,
                    "Y00_balanced_clean": y00,
                    "Y01_balanced_noisy": y01,
                    "Y10_long_tail_clean": y10,
                    "Y11_long_tail_noisy": y11,
                    "long_tail_given_clean": y10 - y00,
                    "noise_given_balanced": y01 - y00,
                    "noise_given_long_tail": y11 - y10,
                    "interaction": y11 - y10 - y01 + y00,
                }
            )
        for class_name in sorted(class_sets[0]):
            for metric in METRICS:
                y00 = condition_payloads["balanced_clean"]["per_class"][class_name][metric]
                y01 = condition_payloads["balanced_noisy"]["per_class"][class_name][metric]
                y10 = condition_payloads["long_tail_clean"]["per_class"][class_name][metric]
                y11 = condition_payloads["long_tail_noisy"]["per_class"][class_name][metric]
                class_effects.append(
                    {
                        "dataset": dataset,
                        "setting": setting,
                        "scenario": scenario,
                        "class_name": class_name,
                        "metric": metric,
                        "Y00_balanced_clean": y00,
                        "Y01_balanced_noisy": y01,
                        "Y10_long_tail_clean": y10,
                        "Y11_long_tail_noisy": y11,
                        "interaction": y11 - y10 - y01 + y00,
                    }
                )

    causal_summary: List[Dict[str, Any]] = []
    for metric in METRICS:
        scenario_rows = [row for row in scenario_effects if row["metric"] == metric]
        class_rows = [row for row in class_effects if row["metric"] == metric]
        summary_row = {
            "metric": metric,
            "num_scenarios": len(scenario_rows),
            "scenario_macro_averaging": "unweighted_arithmetic_mean_over_six_scenarios",
            "Y00_balanced_clean": mean(row["Y00_balanced_clean"] for row in scenario_rows),
            "Y01_balanced_noisy": mean(row["Y01_balanced_noisy"] for row in scenario_rows),
            "Y10_long_tail_clean": mean(row["Y10_long_tail_clean"] for row in scenario_rows),
            "Y11_long_tail_noisy": mean(row["Y11_long_tail_noisy"] for row in scenario_rows),
            "interaction": mean(row["interaction"] for row in scenario_rows),
            "interaction_percent_points": 100.0
            * mean(row["interaction"] for row in scenario_rows),
            "negative_scenario_interactions": sum(row["interaction"] < 0 for row in scenario_rows),
            "num_class_setting_comparisons": len(class_rows),
            "negative_class_interactions": sum(row["interaction"] < 0 for row in class_rows),
            "zero_class_interactions": sum(row["interaction"] == 0 for row in class_rows),
            "positive_class_interactions": sum(row["interaction"] > 0 for row in class_rows),
            "class_interactions_at_most_minus_0_3_points": sum(
                row["interaction"] <= -0.003 for row in class_rows
            ),
            "class_interaction_mean": mean(row["interaction"] for row in class_rows),
            "class_interaction_mean_percent_points": 100.0
            * mean(row["interaction"] for row in class_rows),
        }
        causal_summary.append(summary_row)

    unique_logs = {row["source_log"] for row in cells}
    validation = validate_causal_csvs(
        causal_analysis_dir, cells, class_cells, class_effects, sources
    )
    iauroc_summary = next(row for row in causal_summary if row["metric"] == "I-AUROC")
    causal_json = {
        "image_auroc": iauroc_summary,
        "num_logical_cells": len(cells),
        "num_unique_physical_logs": len(unique_logs),
        "balanced_clean_reuse": {
            "logical_cells": 6,
            "unique_physical_logs": 2,
            "mvtec_log": str(balanced_clean_logs["mvtec"].resolve()),
            "visa_log": str(balanced_clean_logs["visa"].resolve()),
        },
        "reference_csv_validation": validation,
    }
    return cells, class_cells, scenario_effects, class_effects, causal_summary, causal_json


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/home/linux/projects/results/tailguard"),
    )
    parser.add_argument(
        "--causal-analysis-dir",
        type=Path,
        default=repo_root / "analysis" / "tailguard_icassp2027" / "causal_2x2",
    )
    parser.add_argument(
        "--mvtec-balanced-clean-log",
        type=Path,
        default=Path("/mnt/d/DOCUMENTS/Works/IAD/eval/mvtec/log.txt"),
    )
    parser.add_argument(
        "--visa-balanced-clean-log",
        type=Path,
        default=Path("/mnt/d/DOCUMENTS/Works/IAD/eval/visa/log.txt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "analysis_outputs" / "paper_evidence_v1",
    )
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    output_dir = args.output_dir.resolve()
    if results_root == output_dir or results_root in output_dir.parents:
        raise ValueError("Audit output must not be inside the immutable result tree")
    balanced_clean_logs = {
        "mvtec": require_file(args.mvtec_balanced_clean_log),
        "visa": require_file(args.visa_balanced_clean_log),
    }
    sources = SourceRegistry()

    (
        purification_rows,
        purification_macro,
        warmup_rows,
        warmup_macro,
        tail_rows,
        v5_summary,
    ) = audit_v5(results_root, sources)
    (
        causal_cells,
        causal_class_cells,
        causal_scenario_effects,
        causal_class_effects,
        causal_summary,
        causal_json,
    ) = audit_causal(
        results_root,
        balanced_clean_logs,
        args.causal_analysis_dir.resolve(),
        sources,
    )

    cleanup_fields = (
        "dataset",
        "setting",
        "scenario",
        "strategy",
        "selected_iteration",
        "removal_source",
        "num_initial",
        "num_removed",
        "num_contaminated_initial",
        "num_noise_removed",
        "num_clean_initial",
        "num_clean_removed",
        "num_clean_gt_tail_initial",
        "num_clean_gt_tail_removed",
        "removed_set_noise_precision",
        "noise_removal_recall",
        "clean_retention_rate",
        "clean_gt_tail_removal_rate",
        "residual_contamination_rate",
    )
    write_csv(
        output_dir / "purification_matched_count_by_scenario.csv",
        purification_rows,
        cleanup_fields,
    )
    write_csv(
        output_dir / "purification_matched_count_macro.csv",
        purification_macro,
        tuple(purification_macro[0].keys()),
    )
    write_csv(
        output_dir / "warmup_separability_by_scenario.csv",
        warmup_rows,
        tuple(warmup_rows[0].keys()),
    )
    write_csv(
        output_dir / "warmup_separability_macro.csv",
        warmup_macro,
        tuple(warmup_macro[0].keys()),
    )
    write_csv(
        output_dir / "tail_flow_by_scenario.csv",
        tail_rows,
        tuple(tail_rows[0].keys()),
    )
    write_csv(
        output_dir / "tail_flow_summary.csv",
        [v5_summary["tail_flow"]],
        tuple(v5_summary["tail_flow"].keys()),
    )
    write_csv(
        output_dir / "causal_2x2_cells.csv",
        causal_cells,
        tuple(causal_cells[0].keys()),
    )
    write_csv(
        output_dir / "causal_2x2_cells_per_class.csv",
        causal_class_cells,
        tuple(causal_class_cells[0].keys()),
    )
    write_csv(
        output_dir / "causal_2x2_effects_by_scenario.csv",
        causal_scenario_effects,
        tuple(causal_scenario_effects[0].keys()),
    )
    write_csv(
        output_dir / "causal_2x2_effects_per_class.csv",
        causal_class_effects,
        tuple(causal_class_effects[0].keys()),
    )
    write_csv(
        output_dir / "causal_2x2_summary.csv",
        causal_summary,
        tuple(causal_summary[0].keys()),
    )
    write_csv(
        output_dir / "source_files.csv",
        sources.rows(),
        ("path", "size_bytes", "sha256", "roles", "scenarios"),
    )

    limitations = [
        "All recomputed results are seed01; the six scenarios are dataset/imbalance settings, not six random seeds.",
        "The causal table has 24 logical cells but 20 unique physical logs because each dataset's balanced-clean log is reused across its three imbalance settings.",
        "The audit verifies final log metrics, not exact training-command or git-commit parity; those were not embedded in every log.",
        "The causal audit does not re-hash test images. The prior causal audit directly verified MVTec split hashes, while downloaded VisA causal dataset directories were unavailable for the same check.",
        "Purification values are unweighted macro means over six scenarios and do not imply H beats global filtering in every scenario; MVTec Step-K1 has higher global precision and recall.",
        "Matched-count, warm-up, and low-support-candidate diagnostics use injected-contamination and ground-truth-tail labels only after training for analysis; the method does not observe these labels.",
        "Warm-up AUROC/AP covers supported H0 candidates at the selected checkpoint within the recorded early window, not the full training set and not a true early-versus-late comparison.",
    ]
    formulas = {
        "causal_interaction": "Y11 - Y10 - Y01 + Y00",
        "causal_Y00_Y01_Y10_Y11": "balanced-clean, balanced-noisy, long-tail-clean, long-tail-noisy",
        "purification_removed_set_noise_precision": "removed_contaminated / all_removed",
        "purification_noise_removal_recall": "removed_contaminated / all_contaminated",
        "purification_clean_retention_rate": "retained_clean / all_clean",
        "purification_clean_gt_tail_removal_rate": "removed_clean_gt_tail / all_clean_gt_tail",
        "purification_residual_contamination_rate": "retained_contaminated / all_retained",
        "global_matched_count": "sort selected-checkpoint image_score descending, sample_idx ascending; remove exactly |R_H|",
        "macro_averaging": "unweighted arithmetic mean over the six dataset/imbalance scenarios",
        "average_precision": "non-interpolated AP with score ties evaluated as one threshold group (sklearn-compatible)",
    }
    field_provenance = {
        "purification": {
            "labels_and_gt_tail": "sampler_analysis_details.csv: sample_idx,is_contaminated,is_gt_tail",
            "global_score": "selected gbps train_scores.csv: sample_idx,image_score",
            "H_removal": "stage2_removed_samples.csv: sample_idx",
            "selected_iteration": "tailguard_summary.json: gbps_trigger_summary.gbps_selected_iter",
        },
        "warmup": {
            "labels": "sampler_analysis_details.csv: sample_idx,is_contaminated,tail_candidate",
            "scores": "selected h_sample_group_scores.csv: sample_idx,image_score,z_score,p_high",
        },
        "low_support_candidates": {
            "T0_and_labels": "sampler_analysis_details.csv: sample_idx,tail_candidate,is_contaminated,is_gt_tail",
            "H_removal_check": "stage2_removed_samples.csv: sample_idx",
        },
        "causal_2x2": {
            "cell_metrics": "last complete Mean row in each raw log",
            "class_metrics": "last occurrence of each class metric row in each raw log",
            "train_size": "train image number field in each raw log",
        },
    }
    evidence_summary = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "source_result_tree_written": False,
        "v5": v5_summary,
        "causal_2x2": causal_json,
        "formulas": formulas,
        "field_provenance": field_provenance,
        "limitations": limitations,
    }
    write_json(output_dir / "evidence_summary.json", evidence_summary)
    write_json(
        output_dir / "provenance_manifest.json",
        {
            "schema_version": 1,
            "source_result_tree_written": False,
            "source_file_count": len(sources.rows()),
            "source_manifest_csv": str(output_dir / "source_files.csv"),
            "formulas": formulas,
            "field_provenance": field_provenance,
            "limitations": limitations,
        },
    )

    i_auroc = causal_json["image_auroc"]
    print("Wrote paper evidence audit to {}".format(output_dir))
    print(
        "Causal I-AUROC Y00/Y01/Y10/Y11: {:.2f}/{:.2f}/{:.2f}/{:.2f}; "
        "interaction {:.2f} points; negative {}/{}".format(
            100.0 * i_auroc["Y00_balanced_clean"],
            100.0 * i_auroc["Y01_balanced_noisy"],
            100.0 * i_auroc["Y10_long_tail_clean"],
            100.0 * i_auroc["Y11_long_tail_noisy"],
            i_auroc["interaction_percent_points"],
            i_auroc["negative_class_interactions"],
            i_auroc["num_class_setting_comparisons"],
        )
    )
    print(
        "Warm-up AP image/z/p_high: {:.2f}/{:.2f}/{:.2f}".format(
            v5_summary["warmup"]["image_score"]["average_precision_percent"],
            v5_summary["warmup"]["z_score"]["average_precision_percent"],
            v5_summary["warmup"]["p_high"]["average_precision_percent"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
