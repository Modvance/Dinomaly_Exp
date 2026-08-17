#!/usr/bin/env python3
"""Offline evaluation of TailSampler partition repair rules."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


DETAILS_RELATIVE_PATH = Path(
    "tailguard/prepare/tail_sampler_analysis_only/sampler_analysis_details.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/home/linux/projects/results/tailguard/dependency_ablation_v5"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/tailsampler_partition_guard_v1"),
    )
    return parser.parse_args()


def predicted_class_sizes(values: np.ndarray) -> np.ndarray:
    values = np.rint(values).astype(int)
    values = np.maximum(np.sort(values), 1)
    class_sizes: list[int] = []

    while len(values):
        if len(values) < values[0]:
            if class_sizes:
                class_sizes[-1] += len(values)
            else:
                class_sizes.append(len(values))
            break

        count = int(np.rint(values[: values[0]].mean()))
        count = min(count, len(values))
        class_sizes.append(count)
        values = values[count:]

    return np.sort(np.asarray(class_sizes, dtype=float))


def support_mask(df: pd.DataFrame, cutoff: float | None) -> np.ndarray:
    if cutoff is None:
        return np.zeros(len(df), dtype=bool)
    return df["pred_class_size"].to_numpy(dtype=float) <= cutoff


def original_selection(df: pd.DataFrame) -> tuple[np.ndarray, float | None, str]:
    selected = df["is_selected"].to_numpy(dtype=bool)
    cutoff = (
        float(df.loc[selected, "pred_class_size"].max())
        if selected.any()
        else None
    )
    return selected, cutoff, "saved"


def budget_rollback(
    df: pd.DataFrame, budget_ratio: float = 0.15
) -> tuple[np.ndarray, float | None, str]:
    selected, cutoff, _ = original_selection(df)
    budget = math.floor(len(df) * budget_ratio)
    if selected.sum() <= budget:
        return selected, cutoff, "unchanged"

    levels = np.sort(df["pred_class_size"].unique().astype(float))
    safe_levels = [
        level
        for level in levels
        if (cutoff is None or level < cutoff)
        and int((df["pred_class_size"] <= level).sum()) <= budget
    ]
    repaired_cutoff = max(safe_levels) if safe_levels else None
    return support_mask(df, repaired_cutoff), repaired_cutoff, "rollback"


def sample_log_gap(df: pd.DataFrame) -> tuple[np.ndarray, float | None, str]:
    levels = np.sort(df["pred_class_size"].unique().astype(float))
    if len(levels) < 2:
        return np.zeros(len(df), dtype=bool), None, "no_split"
    gap_index = int(np.argmax(np.diff(np.log(levels))))
    cutoff = float(levels[gap_index])
    return support_mask(df, cutoff), cutoff, "global_sample_gap"


def class_log_gap(df: pd.DataFrame) -> tuple[np.ndarray, float | None, str]:
    class_sizes = predicted_class_sizes(df["pred_class_size"].to_numpy())
    if len(class_sizes) < 2:
        return np.zeros(len(df), dtype=bool), None, "no_split"
    gap_index = int(np.argmax(np.diff(np.log(class_sizes))))
    cutoff = float(class_sizes[gap_index])
    return support_mask(df, cutoff), cutoff, "global_class_gap"


def gaussian_log_likelihood(
    values: np.ndarray, weights: np.ndarray, variance_floor: float
) -> float:
    mass = float(weights.sum())
    mean = float(np.sum(weights * values) / mass)
    variance = float(np.sum(weights * (values - mean) ** 2) / mass)
    variance = max(variance, variance_floor)
    return -0.5 * float(
        np.sum(
            weights
            * (
                np.log(2.0 * np.pi * variance)
                + (values - mean) ** 2 / variance
            )
        )
    )


def weighted_bic(df: pd.DataFrame) -> tuple[np.ndarray, float | None, str]:
    counts = df["pred_class_size"].value_counts().sort_index()
    levels = counts.index.to_numpy(dtype=float)
    weights = counts.to_numpy(dtype=float)
    if len(levels) < 2:
        return np.zeros(len(df), dtype=bool), None, "no_split"

    values = np.log(levels)
    total_mass = float(weights.sum())
    global_mean = float(np.sum(weights * values) / total_mass)
    global_variance = float(
        np.sum(weights * (values - global_mean) ** 2) / total_mass
    )
    variance_floor = max(global_variance * 1e-3, 1e-6)
    one_log_likelihood = gaussian_log_likelihood(values, weights, variance_floor)
    one_bic = -2.0 * one_log_likelihood + 2.0 * np.log(total_mass)

    best: tuple[float, int] | None = None
    for split in range(1, len(values)):
        left_mass = float(weights[:split].sum())
        right_mass = float(weights[split:].sum())
        two_log_likelihood = gaussian_log_likelihood(
            values[:split], weights[:split], variance_floor
        )
        two_log_likelihood += gaussian_log_likelihood(
            values[split:], weights[split:], variance_floor
        )
        two_log_likelihood += left_mass * np.log(left_mass / total_mass)
        two_log_likelihood += right_mass * np.log(right_mass / total_mass)
        two_bic = -2.0 * two_log_likelihood + 5.0 * np.log(total_mass)
        if best is None or two_bic < best[0]:
            best = (two_bic, split)

    if best is None or best[0] >= one_bic:
        return np.zeros(len(df), dtype=bool), None, "one_segment"
    cutoff = float(levels[best[1] - 1])
    return support_mask(df, cutoff), cutoff, "two_segment"


def change_point_bic(
    class_sizes: np.ndarray,
) -> tuple[int | None, float | None]:
    values = np.log(np.sort(np.asarray(class_sizes, dtype=float)))
    count = len(values)
    if count < 6:
        return None, None

    one_sse = float(np.sum((values - values.mean()) ** 2))
    epsilon = max(one_sse * 1e-12, 1e-12)
    one_bic = count * math.log(max(one_sse / count, epsilon))
    one_bic += 2 * math.log(count)

    candidates: list[tuple[float, int]] = []
    for split in range(2, count - 1):
        left = values[:split]
        right = values[split:]
        two_sse = float(np.sum((left - left.mean()) ** 2))
        two_sse += float(np.sum((right - right.mean()) ** 2))
        two_bic = count * math.log(max(two_sse / count, epsilon))
        two_bic += 4 * math.log(count)
        candidates.append((two_bic, split))

    two_bic, split = min(candidates)
    return split, one_bic - two_bic


def class_change_bic(
    df: pd.DataFrame,
    evidence_threshold: float = 10.0,
) -> tuple[np.ndarray, float | None, str]:
    class_sizes = predicted_class_sizes(df["pred_class_size"].to_numpy())
    split, delta_bic = change_point_bic(class_sizes)
    if split is None or delta_bic is None:
        return np.zeros(len(df), dtype=bool), None, "abstain_too_few_classes"

    cutoff = float(class_sizes[split - 1])
    if delta_bic <= evidence_threshold:
        return (
            np.zeros(len(df), dtype=bool),
            None,
            f"abstain;candidate_cutoff={cutoff:g};delta_bic={delta_bic:.6g}",
        )
    return (
        support_mask(df, cutoff),
        cutoff,
        f"accepted;delta_bic={delta_bic:.6g}",
    )


def plateau_gap_guard(
    df: pd.DataFrame,
) -> tuple[np.ndarray, float | None, str]:
    selected, original_cutoff, _ = original_selection(df)
    class_sizes = predicted_class_sizes(df["pred_class_size"].to_numpy())
    if original_cutoff is None or len(class_sizes) < 2:
        return selected, original_cutoff, "unchanged"

    gap_index = int(np.argmax(np.diff(np.log(class_sizes))))
    gap_cutoff = float(class_sizes[gap_index])
    plateau_multiplicity = int(np.sum(class_sizes == original_cutoff))

    if plateau_multiplicity >= 2 and original_cutoff > gap_cutoff:
        status = (
            f"guarded;original={original_cutoff:g};"
            f"multiplicity={plateau_multiplicity};gap={gap_cutoff:g}"
        )
        return support_mask(df, gap_cutoff), gap_cutoff, status

    status = (
        f"unchanged;multiplicity={plateau_multiplicity};gap={gap_cutoff:g}"
    )
    return selected, original_cutoff, status


METHODS = {
    "original": original_selection,
    "budget_rollback_15": budget_rollback,
    "sample_log_gap": sample_log_gap,
    "class_log_gap": class_log_gap,
    "weighted_bic": weighted_bic,
    "class_change_bic": class_change_bic,
    "plateau_gap_guard": plateau_gap_guard,
}


def evaluate_selection(df: pd.DataFrame, selected: np.ndarray) -> dict[str, float | int]:
    selected = np.asarray(selected, dtype=bool)
    true_tail = df["is_gt_tail"].to_numpy(dtype=bool)
    contaminated = df["is_contaminated"].to_numpy(dtype=bool)
    true_positive = int(np.sum(selected & true_tail))
    false_positive = int(np.sum(selected & ~true_tail))
    false_negative = int(np.sum(~selected & true_tail))
    selected_count = int(selected.sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        "num_samples": len(df),
        "num_selected": selected_count,
        "selected_ratio": selected_count / len(df),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "selected_contamination": int(np.sum(selected & contaminated)),
        "precision": true_positive / selected_count if selected_count else 0.0,
        "recall": (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        ),
        "f1": 2 * true_positive / denominator if denominator else 0.0,
    }


def main() -> None:
    args = parse_args()
    paths = sorted(args.results_root.glob(f"*_full/{DETAILS_RELATIVE_PATH}"))
    if len(paths) != 30:
        raise RuntimeError(f"expected 30 Full runs, found {len(paths)}")

    rows: list[dict[str, object]] = []
    for path in paths:
        run_name = path.parents[3].name
        scenario = run_name.rsplit("_seed", 1)[0]
        frame = pd.read_csv(path)
        for method_name, method in METHODS.items():
            selected, cutoff, status = method(frame)
            row: dict[str, object] = {
                "run": run_name,
                "scenario": scenario,
                "method": method_name,
                "cutoff": cutoff,
                "status": status,
            }
            row.update(evaluate_selection(frame, selected))
            rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_run = pd.DataFrame(rows)
    per_run.to_csv(args.output_dir / "per_run.csv", index=False)

    overall = (
        per_run.groupby("method")
        .agg(
            runs=("run", "size"),
            selected_total=("num_selected", "sum"),
            true_positive=("true_positive", "sum"),
            false_positive=("false_positive", "sum"),
            selected_contamination=("selected_contamination", "sum"),
            precision_macro=("precision", "mean"),
            recall_macro=("recall", "mean"),
            f1_macro=("f1", "mean"),
            selected_ratio_mean=("selected_ratio", "mean"),
            selected_ratio_std=("selected_ratio", "std"),
            catastrophic_runs=(
                "selected_ratio",
                lambda values: int((values > 0.15 + 1e-12).sum()),
            ),
            abstained_runs=(
                "status",
                lambda values: int(
                    sum(str(value).startswith("abstain") for value in values)
                ),
            ),
        )
        .reset_index()
    )
    overall["precision_micro"] = overall["true_positive"] / (
        overall["true_positive"] + overall["false_positive"]
    )
    overall.to_csv(args.output_dir / "overall_summary.csv", index=False)

    by_scenario = (
        per_run.groupby(["scenario", "method"])
        .agg(
            selected_mean=("num_selected", "mean"),
            selected_std=("num_selected", "std"),
            selected_min=("num_selected", "min"),
            selected_max=("num_selected", "max"),
            selected_ratio_mean=("selected_ratio", "mean"),
            selected_ratio_std=("selected_ratio", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            selected_contamination=("selected_contamination", "sum"),
        )
        .reset_index()
    )
    by_scenario["selected_count_cv"] = by_scenario["selected_std"] / (
        by_scenario["selected_mean"].replace(0, np.nan)
    )
    by_scenario["selected_ratio_cv"] = by_scenario["selected_ratio_std"] / (
        by_scenario["selected_ratio_mean"].replace(0, np.nan)
    )
    by_scenario.to_csv(args.output_dir / "scenario_summary.csv", index=False)

    guard_rows = per_run.loc[
        (per_run["method"] == "plateau_gap_guard")
        & per_run["status"].str.startswith("guarded")
    ]
    guard_rows.to_csv(args.output_dir / "guard_triggered_runs.csv", index=False)

    accepted = per_run.loc[~per_run["status"].str.startswith("abstain")]
    accepted_summary = (
        accepted.groupby("method")
        .agg(
            accepted_runs=("run", "size"),
            selected_total=("num_selected", "sum"),
            true_positive=("true_positive", "sum"),
            false_positive=("false_positive", "sum"),
            selected_contamination=("selected_contamination", "sum"),
            precision_macro=("precision", "mean"),
            recall_macro=("recall", "mean"),
            f1_macro=("f1", "mean"),
            selected_ratio_max=("selected_ratio", "max"),
        )
        .reset_index()
    )
    accepted_summary["precision_micro"] = accepted_summary["true_positive"] / (
        accepted_summary["true_positive"] + accepted_summary["false_positive"]
    )
    accepted_summary.to_csv(args.output_dir / "accepted_summary.csv", index=False)

    rng = np.random.default_rng(20260817)
    stability_rows: list[dict[str, object]] = []
    for path in paths:
        run_name = path.parents[3].name
        frame = pd.read_csv(path, usecols=["pred_class_size"])
        sample_support = frame["pred_class_size"].to_numpy(dtype=float)
        class_sizes = predicted_class_sizes(sample_support)
        base_split, base_delta = change_point_bic(class_sizes)
        if base_split is None or base_delta is None:
            continue

        split_indices: list[int] = []
        delta_values: list[float] = []
        selected_counts: list[int] = []
        for _ in range(1000):
            perturbed_log_sizes = np.log(class_sizes) + rng.normal(
                0.0, 0.05, size=len(class_sizes)
            )
            perturbed_sizes = np.exp(perturbed_log_sizes)
            split, delta_bic = change_point_bic(perturbed_sizes)
            if split is None or delta_bic is None:
                continue
            sorted_sizes = np.sort(perturbed_sizes)
            cutoff = float(sorted_sizes[split - 1])
            split_indices.append(split)
            delta_values.append(delta_bic)
            selected_counts.append(int(np.sum(sample_support <= cutoff)))

        stability_rows.append(
            {
                "run": run_name,
                "num_inferred_classes": len(class_sizes),
                "base_split_index": base_split,
                "base_delta_bic": base_delta,
                "base_accepted": base_delta > 10.0,
                "split_index_agreement": float(
                    np.mean(np.asarray(split_indices) == base_split)
                ),
                "accept_fraction": float(np.mean(np.asarray(delta_values) > 10.0)),
                "delta_bic_q05": float(np.quantile(delta_values, 0.05)),
                "delta_bic_q50": float(np.quantile(delta_values, 0.50)),
                "delta_bic_q95": float(np.quantile(delta_values, 0.95)),
                "selected_q05": float(np.quantile(selected_counts, 0.05)),
                "selected_q50": float(np.quantile(selected_counts, 0.50)),
                "selected_q95": float(np.quantile(selected_counts, 0.95)),
            }
        )
    pd.DataFrame(stability_rows).to_csv(
        args.output_dir / "class_change_bic_stability.csv", index=False
    )

    print(overall.to_string(index=False))
    print("\nGuard-triggered runs")
    print(
        guard_rows[
            [
                "run",
                "cutoff",
                "num_selected",
                "true_positive",
                "false_positive",
                "selected_contamination",
                "precision",
                "recall",
                "status",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
