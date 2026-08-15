#!/usr/bin/env python3
"""Audit and aggregate the six inference-only Routing Shield evaluations.

The script is deliberately independent from the evaluator and training stack:
it uses only Python's standard library, never opens model tensors, and never
writes into the immutable experiment-result tree.  A strict reproduction gate
compares the re-evaluated reconstruction/Full metrics with the original v5
memory-evaluation artifacts before any optimization conclusion is accepted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


SCENARIOS: Tuple[str, ...] = (
    "mvtec_pareto_seed01",
    "mvtec_step_k4_seed01",
    "mvtec_step_k1_seed01",
    "visa_pareto_seed01",
    "visa_step_k4_seed01",
    "visa_step_k1_seed01",
)

METRICS: Tuple[str, ...] = (
    "I-AUROC",
    "I-AP",
    "I-F1",
    "P-AUROC",
    "P-AP",
    "P-F1",
    "P-AUPRO",
)
REPRODUCTION_PER_CLASS_METRICS: Tuple[str, ...] = (
    "I-AUROC",
    "P-AUROC",
    "P-AUPRO",
)
FOCUS_CLASSES = frozenset(("capsule", "pcb1", "fryum"))
VARIANT_ORDER: Tuple[str, ...] = (
    "core",
    "h_only",
    "h_raw_e",
    "full_v5",
    "full_retest",
    "full_shield",
)
EXPECTED_PROTOCOL = {
    "memory_topk_ratio": 0.05,
    "memory_fusion_lambda": 1.0,
    "route_margin_threshold": None,  # JSON-safe representation of -inf.
    "memory_min_class_members": 1,
}
SHIELD_FILES: Tuple[str, ...] = (
    "routing_shield_summary.json",
    "routing_shield_provenance.json",
    "routing_shield_per_class_metrics.csv",
    "routing_shield_route_by_class.csv",
    "routing_shield_route_transitions.csv",
    "routing_shield_scores.csv",
    "routing_shield_prototypes.csv",
)
TRI_STATE_FILES: Tuple[str, ...] = (
    "tri_state_eval_summary.json",
    "tri_state_eval_provenance.json",
    "tri_state_per_class_metrics.csv",
    "tri_state_route_audit.csv",
    "tri_state_scores.csv",
)
TRI_VARIANT_ORDER: Tuple[str, ...] = (
    "core",
    "h_only",
    "h_raw_e",
    "full_v5",
    "full_retest",
    "full_shield",
    "tri_state",
)


def _mean(values: Iterable[float]) -> float:
    numbers = list(values)
    if not numbers:
        raise ValueError("Cannot average an empty sequence")
    return sum(numbers) / len(numbers)


def _float(value: Any, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("{} is not numeric: {!r}".format(context, value)) from error
    if not math.isfinite(number):
        raise ValueError("{} is not finite: {!r}".format(context, value))
    return number


def _int(value: Any, context: str) -> int:
    number = _float(value, context)
    integer = int(number)
    if number != integer:
        raise ValueError("{} is not an integer: {!r}".format(context, value))
    return integer


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object: {}".format(path))
    return payload


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _metric_values(payload: Mapping[str, Any], context: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for metric in METRICS:
        if metric not in payload:
            raise ValueError("{} is missing {}".format(context, metric))
        values[metric] = _float(payload[metric], "{} {}".format(context, metric))
    return values


def _index_per_class(
    rows: Sequence[Mapping[str, Any]], context: str
) -> Dict[Tuple[int, str], Dict[str, float]]:
    indexed: Dict[Tuple[int, str], Dict[str, float]] = {}
    for row in rows:
        class_id = _int(row.get("class_id"), "{} class_id".format(context))
        class_name = str(row.get("class_name", "")).strip()
        if not class_name:
            raise ValueError("{} contains an empty class_name".format(context))
        key = (class_id, class_name)
        if key in indexed:
            raise ValueError("{} contains duplicate class {}".format(context, key))
        indexed[key] = _metric_values(row, "{} class {}".format(context, key))
    if not indexed:
        raise ValueError("{} contains no per-class rows".format(context))
    return indexed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{}.tmp-{}".format(path.name, os.getpid()))
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{}.tmp-{}".format(path.name, os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{}.tmp-{}".format(path.name, os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(str(temporary), str(path))


def required_inputs(
    results_root: Path, shield_root: Path, scenarios: Sequence[str] = SCENARIOS
) -> List[Tuple[str, str, Path]]:
    required: List[Tuple[str, str, Path]] = []
    for scenario in scenarios:
        for variant in ("core", "h_only"):
            required.append((scenario, "{}_summary".format(variant), results_root / (
                "{}_{}".format(scenario, variant)
            ) / "tailguard" / "tailguard_summary.json"))
        for variant in ("h_raw_e", "full"):
            required.append((scenario, "{}_memory_eval".format(variant), results_root / (
                "{}_{}".format(scenario, variant)
            ) / "tailguard" / "memory" / "memory_eval_summary.json"))
        for filename in SHIELD_FILES:
            required.append((scenario, filename, shield_root / scenario / filename))
    return required


def missing_inputs(
    results_root: Path, shield_root: Path, scenarios: Sequence[str] = SCENARIOS
) -> List[Dict[str, str]]:
    return [
        {"scenario": scenario, "role": role, "path": str(path.resolve())}
        for scenario, role, path in required_inputs(results_root, shield_root, scenarios)
        if not path.is_file()
    ]


def required_tri_state_inputs(
    results_root: Path, tri_state_root: Path, scenarios: Sequence[str] = SCENARIOS
) -> List[Tuple[str, str, Path]]:
    required: List[Tuple[str, str, Path]] = []
    for scenario in scenarios:
        for variant in ("core", "h_only"):
            required.append((scenario, "{}_summary".format(variant), results_root / (
                "{}_{}".format(scenario, variant)
            ) / "tailguard" / "tailguard_summary.json"))
        for variant in ("h_raw_e", "full"):
            required.append((scenario, "{}_memory_eval".format(variant), results_root / (
                "{}_{}".format(scenario, variant)
            ) / "tailguard" / "memory" / "memory_eval_summary.json"))
        for filename in TRI_STATE_FILES:
            required.append((scenario, filename, tri_state_root / scenario / filename))
    return required


def missing_tri_state_inputs(
    results_root: Path, tri_state_root: Path, scenarios: Sequence[str] = SCENARIOS
) -> List[Dict[str, str]]:
    return [
        {"scenario": scenario, "role": role, "path": str(path.resolve())}
        for scenario, role, path in required_tri_state_inputs(
            results_root, tri_state_root, scenarios
        )
        if not path.is_file()
    ]


class SourceManifest:
    def __init__(self) -> None:
        self._rows: MutableMapping[str, Dict[str, Any]] = {}

    def add(self, path: Path, scenario: str, role: str) -> Path:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError("Required audit input is missing: {}".format(resolved))
        key = str(resolved)
        if key not in self._rows:
            stat = resolved.stat()
            self._rows[key] = {
                "path": key,
                "size_bytes": stat.st_size,
                "sha256": _sha256(resolved),
                "scenarios": set(),
                "roles": set(),
            }
        self._rows[key]["scenarios"].add(scenario)
        self._rows[key]["roles"].add(role)
        return resolved

    def rows(self) -> List[Dict[str, Any]]:
        output = []
        for key in sorted(self._rows):
            row = self._rows[key]
            output.append({
                "path": row["path"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
                "scenarios": ";".join(sorted(row["scenarios"])),
                "roles": ";".join(sorted(row["roles"])),
            })
        return output


def _validate_protocol(summary: Mapping[str, Any], provenance: Mapping[str, Any], scenario: str) -> None:
    protocol = summary.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("{} summary has no protocol object".format(scenario))
    for key, expected in EXPECTED_PROTOCOL.items():
        observed = protocol.get(key)
        if expected is None:
            if observed is not None:
                raise ValueError(
                    "{} protocol {} must encode -inf as null, got {!r}".format(
                        scenario, key, observed
                    )
                )
        elif _float(observed, "{} protocol {}".format(scenario, key)) != expected:
            raise ValueError("{} protocol {} differs from original v5".format(scenario, key))
    if summary.get("no_training") is not True or provenance.get("no_training") is not True:
        raise ValueError("{} is not marked as inference-only".format(scenario))
    if summary.get("evaluator") != "dinomaly_tailguard_routing_shield_eval":
        raise ValueError("{} has an unexpected evaluator".format(scenario))
    contract = provenance.get("single_pass_contract", {})
    for key in ("shared_full_checkpoint", "shared_retained_train_set", "one_test_joint_model_pass"):
        if contract.get(key) is not True:
            raise ValueError("{} provenance violates {}".format(scenario, key))
    equivalence = contract.get("equivalence_audit")
    if not isinstance(equivalence, dict):
        raise ValueError("{} provenance has no equivalence audit".format(scenario))
    for key in ("max_abs_reconstruction_map_error", "max_abs_patch_feature_error"):
        error = _float(equivalence.get(key), "{} equivalence {}".format(scenario, key))
        if error > 1e-5:
            raise ValueError("{} equivalence {} exceeds 1e-5: {}".format(scenario, key, error))


def _add_gate_comparison(
    rows: List[Dict[str, Any]],
    scenario: str,
    scope: str,
    mode: str,
    metric: str,
    old_value: float,
    new_value: float,
    tolerance: float,
    class_id: Optional[int] = None,
    class_name: str = "",
) -> None:
    difference = new_value - old_value
    rows.append({
        "scenario": scenario,
        "scope": scope,
        "mode": mode,
        "class_id": "" if class_id is None else class_id,
        "class_name": class_name,
        "metric": metric,
        "old_value": old_value,
        "new_value": new_value,
        "signed_diff": difference,
        "abs_diff": abs(difference),
        "tolerance": tolerance,
        "passed": int(abs(difference) <= tolerance),
    })


def _gate_tolerance(metric: str, default_tolerance: float, aupro_tolerance: float) -> float:
    # PRO uses a discretized numerical integration; tiny platform-dependent
    # differences are larger than those of rank metrics despite equal inputs.
    return aupro_tolerance if metric == "P-AUPRO" else default_tolerance


def _markdown_percent(value: float) -> str:
    return "{:.3f}".format(100.0 * value)


def _render_report(
    gate_passed: bool,
    tolerance: float,
    macro_lookup: Mapping[Tuple[str, str], float],
    routing_rows: Sequence[Mapping[str, Any]],
    focus_rows: Sequence[Mapping[str, Any]],
    assessment: Optional[Mapping[str, Any]],
    failed_gate_rows: Sequence[Mapping[str, Any]],
    scenarios: Sequence[str],
    transition_rows: Sequence[Mapping[str, Any]],
    per_class_delta_rows: Sequence[Mapping[str, Any]],
    aupro_tolerance: float,
) -> str:
    is_complete = tuple(scenarios) == SCENARIOS
    title = "六场景" if is_complete else "阶段性（{} 场景）".format(len(scenarios))
    lines = ["# Routing Shield {}结果审计".format(title), ""]
    if gate_passed:
        lines.extend([
            (
                "**复现闸门：GREEN / PASSED。** 下面的优化比较可用于实验决策。"
                if is_complete else
                "**复现闸门：GREEN / PASSED。** 当前仅为阶段性结果，不形成六场景优化结论。"
            ),
            "",
        ])
    else:
        lines.extend([
            "**复现闸门：RED / BLOCKED。** 重测未严格复现原 v5，下面不得形成优化结论。",
            "",
            "失败项数量：{}；一般指标绝对容差：`{}`；P-AUPRO 绝对容差：`{}`。".format(
                len(failed_gate_rows), tolerance, aupro_tolerance
            ),
            "",
        ])
        for row in failed_gate_rows[:20]:
            lines.append(
                "- `{scenario}` `{scope}` `{mode}` `{class_name}` `{metric}`：差值 `{signed_diff:.9g}`".format(
                    **row
                )
            )
        lines.append("")

    lines.extend([
        "复现闸门对 P-AUPRO 使用 `5e-6` 容差，仅用于容纳 PRO 离散数值积分的微小平台差异；其余指标默认使用 `1e-6`。",
        "",
    ])

    lines.extend([
        "## {}等权平均".format(title),
        "",
        "数值单位为百分数；{}个场景等权平均。".format(len(scenarios)),
        "",
        "| 变体 | I-AUROC | P-AUROC | P-AUPRO |",
        "|---|---:|---:|---:|",
    ])
    labels = {
        "core": "Unified Reconstruction Core",
        "h_only": "+ H",
        "h_raw_e": "+ H + Raw-E",
        "full_v5": "原 Full",
        "full_retest": "重测 Full",
        "full_shield": "Full + Routing Shield",
    }
    for variant in VARIANT_ORDER:
        lines.append(
            "| {} | {} | {} | {} |".format(
                labels[variant],
                _markdown_percent(macro_lookup[(variant, "I-AUROC")]),
                _markdown_percent(macro_lookup[(variant, "P-AUROC")]),
                _markdown_percent(macro_lookup[(variant, "P-AUPRO")]),
            )
        )
    lines.append("")

    if gate_passed and assessment is not None and is_complete:
        lines.extend([
            "## 判定",
            "",
            "结论：**{}**。".format(assessment["verdict"]),
            "",
            "- 相对原 Full：I-AUROC {:+.3f}，P-AUROC {:+.3f}，P-AUPRO {:+.3f} 个百分点。".format(
                100.0 * assessment["delta_vs_full"]["I-AUROC"],
                100.0 * assessment["delta_vs_full"]["P-AUROC"],
                100.0 * assessment["delta_vs_full"]["P-AUPRO"],
            ),
            "- 相对 H+Raw-E：I-AUROC {:+.3f}，P-AUROC {:+.3f}，P-AUPRO {:+.3f} 个百分点。".format(
                100.0 * assessment["delta_vs_h_raw_e"]["I-AUROC"],
                100.0 * assessment["delta_vs_h_raw_e"]["P-AUROC"],
                100.0 * assessment["delta_vs_h_raw_e"]["P-AUPRO"],
            ),
            "- 相对 H-only：I-AUROC {:+.3f}，P-AUROC {:+.3f}，P-AUPRO {:+.3f} 个百分点。".format(
                100.0 * assessment["delta_vs_h_only"]["I-AUROC"],
                100.0 * assessment["delta_vs_h_only"]["P-AUROC"],
                100.0 * assessment["delta_vs_h_only"]["P-AUPRO"],
            ),
            "",
        ])

    total_images = sum(int(row["num_test_images"]) for row in routing_rows)
    total_wins = sum(int(row["num_shield_wins"]) for row in routing_rows)
    total_applied = sum(int(row["num_full_memory_applied"]) for row in routing_rows)
    total_blocked = sum(int(row["num_full_memory_blocked"]) for row in routing_rows)
    wins_without_block = total_wins - total_blocked
    lines.extend([
        "## 路由行为",
        "",
        "- 当前 {} 个场景共 {} 张测试图，shield 命中 {} 张（{:.2f}%）。".format(
            len(scenarios),
            total_images, total_wins, 100.0 * total_wins / total_images if total_images else 0.0
        ),
        "- 原 Full 对 {} 张图应用 memory，其中 {} 张被 shield 阻断（{:.2f}%）。".format(
            total_applied,
            total_blocked,
            100.0 * total_blocked / total_applied if total_applied else 0.0,
        ),
        "- 另有 {} 次 shield 命中未阻断 memory（原路由不具备 memory），因此不改变最终分数。".format(
            wins_without_block
        ),
        "",
    ])

    if focus_rows:
        lines.extend([
            "## 重点类别",
            "",
            "| 场景 | 类别 | Shield- Full I-AUROC | Shield- Raw-E I-AUROC | 命中 | 阻断 | 正常/异常阻断 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for row in focus_rows:
            lines.append(
                "| {scenario} | {class_name} | {delta_i_auroc_vs_full_pp:+.3f} | "
                "{delta_i_auroc_vs_raw_e_pp:+.3f} | {num_shield_wins} | "
                "{num_full_memory_blocked} | {num_normal_memory_blocked}/{num_anomaly_memory_blocked} |".format(
                    **row
                )
            )
        lines.append("")

        capsule_rows = [
            row for row in focus_rows
            if row["class_name"].lower() == "capsule"
            and int(row["num_full_memory_blocked"]) > 0
        ]
        if capsule_rows:
            capsule = capsule_rows[0]
            if abs(float(capsule["shield_i_auroc"]) - float(capsule["recon_i_auroc"])) <= tolerance:
                lines.extend([
                    "`capsule` 的 Shield I-AUROC 与 reconstruction 相同（{:.3f}%）。这说明 Shield "
                    "成功消除了错误 memory，但只执行回退，未恢复 Raw-E 中 capsule 专属 memory 的收益。".format(
                        100.0 * float(capsule["shield_i_auroc"])
                    ),
                    "",
                ])

    changed = []
    for row in per_class_delta_rows:
        if row["metric"] in REPRODUCTION_PER_CLASS_METRICS and abs(float(row["shield_minus_full"])) > 1e-12:
            changed.append(row)
    if changed:
        lines.extend([
            "## 指标变化定位",
            "",
            "以下为 Shield 相对重测 Full 发生实际变化的类别与三项主指标：",
            "",
            "| 场景 | 类别 | 指标 | 变化（百分点） |",
            "|---|---|---|---:|",
        ])
        for row in changed:
            lines.append(
                "| {scenario} | {class_name} | {metric} | {delta:+.3f} |".format(
                    scenario=row["scenario"],
                    class_name=row["class_name"],
                    metric=row["metric"],
                    delta=100.0 * float(row["shield_minus_full"]),
                )
            )
        lines.append("")

    blocked_transitions = [
        row for row in transition_rows if int(row["num_full_memory_blocked"]) > 0
    ]
    if blocked_transitions:
        lines.extend([
            "## 被阻断的路由",
            "",
            "| 场景 | 原 Full memory 类 | Shield 来源类 | 阻断 | 正常/异常 |",
            "|---|---|---|---:|---:|",
        ])
        for row in blocked_transitions:
            lines.append(
                "| {scenario} | {full_route_member_class_names} | {shield_source_class_name} | "
                "{num_full_memory_blocked} | {num_normal}/{num_anomaly} |".format(**row)
            )
        lines.append("")
    lines.extend([
        "完整的逐场景、逐类、复现闸门和路由明细见同目录 CSV/JSON。",
        "",
    ])
    return "\n".join(lines)


def audit(
    results_root: Path,
    shield_root: Path,
    output_dir: Path,
    tolerance: float,
    scenarios: Sequence[str] = SCENARIOS,
    aupro_tolerance: float = 5e-6,
) -> Dict[str, Any]:
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite non-negative number")
    if not math.isfinite(aupro_tolerance) or aupro_tolerance < 0:
        raise ValueError("aupro_tolerance must be a finite non-negative number")
    scenarios = tuple(scenarios)
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("scenarios must be non-empty and unique")
    unknown = sorted(set(scenarios).difference(SCENARIOS))
    if unknown:
        raise ValueError("Unknown scenarios: {}".format(unknown))
    missing = missing_inputs(results_root, shield_root, scenarios)
    if missing:
        raise FileNotFoundError(
            "{} required inputs are missing; first missing input: {}".format(
                len(missing), missing[0]["path"]
            )
        )

    sources = SourceManifest()
    scenario_metric_rows: List[Dict[str, Any]] = []
    per_class_delta_rows: List[Dict[str, Any]] = []
    reproduction_rows: List[Dict[str, Any]] = []
    routing_rows: List[Dict[str, Any]] = []
    routing_class_rows: List[Dict[str, Any]] = []
    transition_rows: List[Dict[str, Any]] = []
    focus_rows: List[Dict[str, Any]] = []

    for scenario in scenarios:
        old: Dict[str, Dict[str, Any]] = {}
        for variant in ("core", "h_only"):
            path = results_root / ("{}_{}".format(scenario, variant)) / "tailguard" / "tailguard_summary.json"
            old[variant] = _read_json(sources.add(path, scenario, "{}_summary".format(variant)))
        for variant in ("h_raw_e", "full"):
            path = results_root / ("{}_{}".format(scenario, variant)) / "tailguard" / "memory" / "memory_eval_summary.json"
            old[variant] = _read_json(sources.add(path, scenario, "{}_memory_eval".format(variant)))

        shield_dir = shield_root / scenario
        shield_summary = _read_json(sources.add(
            shield_dir / "routing_shield_summary.json", scenario, "shield_summary"
        ))
        shield_provenance = _read_json(sources.add(
            shield_dir / "routing_shield_provenance.json", scenario, "shield_provenance"
        ))
        per_class_csv = _read_csv(sources.add(
            shield_dir / "routing_shield_per_class_metrics.csv", scenario, "shield_per_class"
        ))
        route_class_csv = _read_csv(sources.add(
            shield_dir / "routing_shield_route_by_class.csv", scenario, "shield_routes_by_class"
        ))
        transitions_csv = _read_csv(sources.add(
            shield_dir / "routing_shield_route_transitions.csv", scenario, "shield_transitions"
        ))
        score_csv = _read_csv(sources.add(
            shield_dir / "routing_shield_scores.csv", scenario, "shield_scores"
        ))
        shield_prototypes_csv = _read_csv(sources.add(
            shield_dir / "routing_shield_prototypes.csv", scenario, "shield_prototypes"
        ))
        _validate_protocol(shield_summary, shield_provenance, scenario)
        expected_checkpoint = (
            results_root / (scenario + "_full") / "final_model.pt"
        ).resolve()
        observed_checkpoint = Path(
            str(shield_provenance.get("checkpoint", {}).get("path", ""))
        ).resolve()
        if observed_checkpoint != expected_checkpoint:
            raise ValueError(
                "{} provenance checkpoint differs: {} != {}".format(
                    scenario, observed_checkpoint, expected_checkpoint
                )
            )

        core_metrics = _metric_values(old["core"]["final_eval_summary"], scenario + " core")
        h_metrics = _metric_values(old["h_only"]["final_eval_summary"], scenario + " h_only")
        raw_metrics = _metric_values(old["h_raw_e"]["summary"], scenario + " h_raw_e")
        full_metrics = _metric_values(old["full"]["summary"], scenario + " full_v5")
        retest_metrics = _metric_values(shield_summary["full"], scenario + " full_retest")
        shield_metrics = _metric_values(shield_summary["full_shield"], scenario + " full_shield")
        recon_metrics = _metric_values(shield_summary["reconstruction"], scenario + " recon_retest")
        values_by_variant = {
            "core": core_metrics,
            "h_only": h_metrics,
            "h_raw_e": raw_metrics,
            "full_v5": full_metrics,
            "full_retest": retest_metrics,
            "full_shield": shield_metrics,
        }
        for variant, values in values_by_variant.items():
            scenario_metric_rows.append({"scenario": scenario, "variant": variant, **values})

        old_full_modes = old["full"].get("metrics_by_mode", {})
        if "recon" not in old_full_modes:
            raise ValueError("{} old Full has no reconstruction metrics".format(scenario))
        old_recon_metrics = _metric_values(
            old_full_modes["recon"]["summary"], scenario + " old full recon"
        )
        for mode, prior, current in (
            ("recon", old_recon_metrics, recon_metrics),
            ("full", full_metrics, retest_metrics),
        ):
            for metric in METRICS:
                _add_gate_comparison(
                    reproduction_rows,
                    scenario,
                    "summary",
                    mode,
                    metric,
                    prior[metric],
                    current[metric],
                    _gate_tolerance(metric, tolerance, aupro_tolerance),
                )

        new_by_mode: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in per_class_csv:
            mode = str(row.get("mode", ""))
            new_by_mode[mode].append(row)
        expected_modes = {"recon", "full", "full_shield"}
        if set(new_by_mode) != expected_modes:
            raise ValueError(
                "{} has unexpected per-class modes: {}".format(scenario, sorted(new_by_mode))
            )
        new_class = {
            mode: _index_per_class(rows, "{} new {}".format(scenario, mode))
            for mode, rows in new_by_mode.items()
        }
        old_full_class = _index_per_class(
            old["full"]["per_class_metrics"], scenario + " old Full"
        )
        old_recon_class = _index_per_class(
            old_full_modes["recon"]["per_class_metrics"], scenario + " old recon"
        )
        old_raw_class = _index_per_class(
            old["h_raw_e"]["per_class_metrics"], scenario + " old Raw-E"
        )
        class_keys = set(old_full_class)
        for name, indexed in (
            ("old recon", old_recon_class),
            ("old Raw-E", old_raw_class),
            ("new recon", new_class["recon"]),
            ("new Full", new_class["full"]),
            ("new Full+Shield", new_class["full_shield"]),
        ):
            if set(indexed) != class_keys:
                raise ValueError("{} {} class identities differ".format(scenario, name))

        for mode, prior, current in (
            ("recon", old_recon_class, new_class["recon"]),
            ("full", old_full_class, new_class["full"]),
        ):
            for class_id, class_name in sorted(class_keys):
                for metric in REPRODUCTION_PER_CLASS_METRICS:
                    _add_gate_comparison(
                        reproduction_rows,
                        scenario,
                        "per_class",
                        mode,
                        metric,
                        prior[(class_id, class_name)][metric],
                        current[(class_id, class_name)][metric],
                        _gate_tolerance(metric, tolerance, aupro_tolerance),
                        class_id,
                        class_name,
                    )

        route_by_key: Dict[Tuple[int, str], Dict[str, Any]] = {}
        for source_row in route_class_csv:
            class_id = _int(source_row.get("class_id"), scenario + " route class_id")
            class_name = str(source_row.get("class_name", "")).strip()
            key = (class_id, class_name)
            if key in route_by_key:
                raise ValueError("{} has duplicate route class {}".format(scenario, key))
            row: Dict[str, Any] = {"scenario": scenario, "class_id": class_id, "class_name": class_name}
            for field in (
                "num_images",
                "num_shield_wins",
                "num_full_memory_applied",
                "num_full_memory_blocked",
                "num_normal_memory_blocked",
                "num_anomaly_memory_blocked",
            ):
                row[field] = _int(source_row.get(field), "{} route {}".format(scenario, field))
            row["shield_win_ratio"] = _float(source_row.get("shield_win_ratio"), scenario + " shield_win_ratio")
            row["blocked_among_full_applied_ratio"] = _float(
                source_row.get("blocked_among_full_applied_ratio"),
                scenario + " blocked_among_full_applied_ratio",
            )
            route_by_key[key] = row
            routing_class_rows.append(row)
        if set(route_by_key) != class_keys:
            raise ValueError("{} route/per-class identities differ".format(scenario))

        item_list = shield_provenance.get("dataset", {}).get("item_list")
        if not isinstance(item_list, list) or not all(isinstance(item, str) for item in item_list):
            raise ValueError("{} provenance has no valid item_list".format(scenario))
        shield_source_names: Dict[int, str] = {}
        for row in shield_prototypes_csv:
            shield_id = _int(row.get("shield_id"), scenario + " shield_id")
            source_class_id = _int(row.get("class_id"), scenario + " shield source class_id")
            if source_class_id < 0 or source_class_id >= len(item_list):
                raise ValueError("{} shield source class_id is out of range".format(scenario))
            if str(row.get("has_patch_memory", "")).strip().lower() not in ("false", "0"):
                raise ValueError("{} shield prototype unexpectedly has patch memory".format(scenario))
            shield_source_names[shield_id] = item_list[source_class_id]
        if len(shield_source_names) != len(shield_prototypes_csv):
            raise ValueError("{} contains duplicate shield ids".format(scenario))
        expected_shields = _int(
            shield_summary.get("routing", {}).get("num_shield_prototypes"),
            scenario + " num_shield_prototypes",
        )
        provenance_shields = _int(
            shield_provenance.get("tail_attached_samples", {}).get("num_samples"),
            scenario + " tail-attached samples",
        )
        if len(shield_source_names) != expected_shields or expected_shields != provenance_shields:
            raise ValueError("{} shield prototype counts differ".format(scenario))
        registry_members_path = (
            results_root / (scenario + "_full") / "tailguard" / "pseudoclasses"
            / "pseudo_class_members.csv"
        )
        registry_members = _read_csv(sources.add(
            registry_members_path, scenario, "full_registry_members"
        ))
        route_member_names: Dict[int, set] = defaultdict(set)
        for row in registry_members:
            pseudo_class_id = _int(row.get("pseudo_class_id"), scenario + " pseudo_class_id")
            member_class_id = _int(row.get("class_id"), scenario + " member class_id")
            if member_class_id < 0 or member_class_id >= len(item_list):
                raise ValueError("{} registry class_id is out of range".format(scenario))
            route_member_names[pseudo_class_id].add(item_list[member_class_id])
        for row in transitions_csv:
            full_route_id = _int(
                row.get("full_predicted_pseudo_class_id"), scenario + " transition full route"
            )
            shield_id = _int(row.get("shield_id"), scenario + " transition shield_id")
            transition_rows.append({
                "scenario": scenario,
                "full_predicted_pseudo_class_id": full_route_id,
                "full_predicted_pseudo_class_type": row.get("full_predicted_pseudo_class_type", ""),
                "full_route_member_class_names": ";".join(sorted(route_member_names[full_route_id])),
                "shield_route_kind": row.get("shield_route_kind", ""),
                "shield_id": shield_id,
                "shield_source_class_name": shield_source_names.get(shield_id, ""),
                "shield_source_sample_idx": _int(
                    row.get("shield_source_sample_idx"), scenario + " transition source sample"
                ),
                "num_images": _int(row.get("num_images"), scenario + " transition images"),
                "num_normal": _int(row.get("num_normal"), scenario + " transition normal"),
                "num_anomaly": _int(row.get("num_anomaly"), scenario + " transition anomaly"),
                "num_full_memory_blocked": _int(
                    row.get("num_full_memory_blocked"), scenario + " transition blocked"
                ),
            })

        # Recompute route counts from image-level records, then require exact
        # agreement with both evaluator summaries before reporting them.
        score_counts = {
            "num_test_images": len(score_csv),
            "num_shield_wins": sum(_int(row["shield_win"], scenario + " shield_win") for row in score_csv),
            "num_full_memory_applied": sum(_int(row["full_mem_applied"], scenario + " full_mem_applied") for row in score_csv),
            "num_full_memory_blocked": sum(_int(row["full_memory_blocked"], scenario + " blocked") for row in score_csv),
        }
        route_counts = {
            "num_test_images": sum(row["num_images"] for row in route_by_key.values()),
            "num_shield_wins": sum(row["num_shield_wins"] for row in route_by_key.values()),
            "num_full_memory_applied": sum(row["num_full_memory_applied"] for row in route_by_key.values()),
            "num_full_memory_blocked": sum(row["num_full_memory_blocked"] for row in route_by_key.values()),
        }
        summary_routing = shield_summary.get("routing", {})
        for field, value in score_counts.items():
            if route_counts[field] != value or _int(summary_routing.get(field), scenario + " summary " + field) != value:
                raise ValueError("{} route count mismatch for {}".format(scenario, field))
        routing_rows.append({
            "scenario": scenario,
            **score_counts,
            "shield_win_ratio": (
                score_counts["num_shield_wins"] / score_counts["num_test_images"]
                if score_counts["num_test_images"] else 0.0
            ),
            "blocked_among_full_applied_ratio": (
                score_counts["num_full_memory_blocked"] / score_counts["num_full_memory_applied"]
                if score_counts["num_full_memory_applied"] else 0.0
            ),
            "num_normal_memory_blocked": sum(
                row["num_normal_memory_blocked"] for row in route_by_key.values()
            ),
            "num_anomaly_memory_blocked": sum(
                row["num_anomaly_memory_blocked"] for row in route_by_key.values()
            ),
        })

        for class_id, class_name in sorted(class_keys):
            for metric in METRICS:
                shield_value = new_class["full_shield"][(class_id, class_name)][metric]
                full_value = new_class["full"][(class_id, class_name)][metric]
                raw_value = old_raw_class[(class_id, class_name)][metric]
                recon_value = new_class["recon"][(class_id, class_name)][metric]
                per_class_delta_rows.append({
                    "scenario": scenario,
                    "class_id": class_id,
                    "class_name": class_name,
                    "metric": metric,
                    "recon_retest": recon_value,
                    "full_v5": old_full_class[(class_id, class_name)][metric],
                    "full_retest": full_value,
                    "h_raw_e": raw_value,
                    "full_shield": shield_value,
                    "shield_minus_full": shield_value - full_value,
                    "shield_minus_h_raw_e": shield_value - raw_value,
                    "shield_minus_recon": shield_value - recon_value,
                })
            if class_name.lower() in FOCUS_CLASSES:
                route = route_by_key[(class_id, class_name)]
                focus_rows.append({
                    "scenario": scenario,
                    "class_id": class_id,
                    "class_name": class_name,
                    "recon_i_auroc": new_class["recon"][(class_id, class_name)]["I-AUROC"],
                    "raw_e_i_auroc": old_raw_class[(class_id, class_name)]["I-AUROC"],
                    "full_i_auroc": new_class["full"][(class_id, class_name)]["I-AUROC"],
                    "shield_i_auroc": new_class["full_shield"][(class_id, class_name)]["I-AUROC"],
                    "delta_i_auroc_vs_full_pp": 100.0 * (
                        new_class["full_shield"][(class_id, class_name)]["I-AUROC"]
                        - new_class["full"][(class_id, class_name)]["I-AUROC"]
                    ),
                    "delta_i_auroc_vs_raw_e_pp": 100.0 * (
                        new_class["full_shield"][(class_id, class_name)]["I-AUROC"]
                        - old_raw_class[(class_id, class_name)]["I-AUROC"]
                    ),
                    "full_p_auroc": new_class["full"][(class_id, class_name)]["P-AUROC"],
                    "shield_p_auroc": new_class["full_shield"][(class_id, class_name)]["P-AUROC"],
                    "full_p_aupro": new_class["full"][(class_id, class_name)]["P-AUPRO"],
                    "shield_p_aupro": new_class["full_shield"][(class_id, class_name)]["P-AUPRO"],
                    **{key: route[key] for key in (
                        "num_images",
                        "num_shield_wins",
                        "num_full_memory_applied",
                        "num_full_memory_blocked",
                        "num_normal_memory_blocked",
                        "num_anomaly_memory_blocked",
                    )},
                })

    macro_rows: List[Dict[str, Any]] = []
    macro_lookup: Dict[Tuple[str, str], float] = {}
    for variant in VARIANT_ORDER:
        variant_rows = [row for row in scenario_metric_rows if row["variant"] == variant]
        if len(variant_rows) != len(scenarios):
            raise RuntimeError("Variant {} does not have all requested scenario rows".format(variant))
        for metric in METRICS:
            value = _mean(float(row[metric]) for row in variant_rows)
            macro_lookup[(variant, metric)] = value
            macro_rows.append({"variant": variant, "metric": metric, "macro_value": value})

    failed_gate_rows = [row for row in reproduction_rows if not row["passed"]]
    gate_passed = not failed_gate_rows
    max_gate_diff = max(float(row["abs_diff"]) for row in reproduction_rows)
    assessment: Optional[Dict[str, Any]] = None
    if gate_passed and tuple(scenarios) == SCENARIOS:
        delta_vs_full = {
            metric: macro_lookup[("full_shield", metric)] - macro_lookup[("full_v5", metric)]
            for metric in METRICS
        }
        delta_vs_raw = {
            metric: macro_lookup[("full_shield", metric)] - macro_lookup[("h_raw_e", metric)]
            for metric in METRICS
        }
        delta_vs_h = {
            metric: macro_lookup[("full_shield", metric)] - macro_lookup[("h_only", metric)]
            for metric in METRICS
        }
        checks = {
            "improves_i_auroc_vs_full": delta_vs_full["I-AUROC"] > tolerance,
            "reaches_h_raw_e_i_auroc": delta_vs_raw["I-AUROC"] >= -tolerance,
            "preserves_full_p_auroc": delta_vs_full["P-AUROC"] >= -tolerance,
            "preserves_full_p_aupro": delta_vs_full["P-AUPRO"] >= -tolerance,
        }
        if all(checks.values()):
            verdict = "SUCCESS：追平 Raw-E 图像指标，并保留原 Full 的像素优势"
        elif checks["improves_i_auroc_vs_full"]:
            verdict = "PARTIAL：修复有所收益，但未同时满足全部目标"
        else:
            verdict = "NO-GAIN：Routing Shield 未改善六场景图像宏平均"
        assessment = {
            "verdict": verdict,
            "checks": checks,
            "delta_vs_full": delta_vs_full,
            "delta_vs_h_raw_e": delta_vs_raw,
            "delta_vs_h_only": delta_vs_h,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "reproduction_gate.csv",
        reproduction_rows,
        (
            "scenario", "scope", "mode", "class_id", "class_name", "metric",
            "old_value", "new_value", "signed_diff", "abs_diff", "tolerance", "passed",
        ),
    )
    _write_csv(
        output_dir / "routing_transitions_resolved.csv",
        transition_rows,
        (
            "scenario", "full_predicted_pseudo_class_id", "full_predicted_pseudo_class_type",
            "full_route_member_class_names", "shield_route_kind", "shield_id",
            "shield_source_class_name", "shield_source_sample_idx", "num_images",
            "num_normal", "num_anomaly", "num_full_memory_blocked",
        ),
    )
    _write_csv(
        output_dir / "scenario_metrics.csv",
        scenario_metric_rows,
        ("scenario", "variant", *METRICS),
    )
    _write_csv(
        output_dir / "macro_metrics.csv",
        macro_rows,
        ("variant", "metric", "macro_value"),
    )
    _write_csv(
        output_dir / "per_class_deltas.csv",
        per_class_delta_rows,
        (
            "scenario", "class_id", "class_name", "metric", "recon_retest", "full_v5",
            "full_retest", "h_raw_e", "full_shield", "shield_minus_full",
            "shield_minus_h_raw_e", "shield_minus_recon",
        ),
    )
    _write_csv(
        output_dir / "routing_by_scenario.csv",
        routing_rows,
        (
            "scenario", "num_test_images", "num_shield_wins", "shield_win_ratio",
            "num_full_memory_applied", "num_full_memory_blocked",
            "blocked_among_full_applied_ratio", "num_normal_memory_blocked",
            "num_anomaly_memory_blocked",
        ),
    )
    _write_csv(
        output_dir / "routing_by_class.csv",
        routing_class_rows,
        (
            "scenario", "class_id", "class_name", "num_images", "num_shield_wins",
            "shield_win_ratio", "num_full_memory_applied", "num_full_memory_blocked",
            "blocked_among_full_applied_ratio", "num_normal_memory_blocked",
            "num_anomaly_memory_blocked",
        ),
    )
    _write_csv(
        output_dir / "focus_capsule_pcb1_fryum.csv",
        focus_rows,
        (
            "scenario", "class_id", "class_name", "recon_i_auroc", "raw_e_i_auroc",
            "full_i_auroc", "shield_i_auroc", "delta_i_auroc_vs_full_pp",
            "delta_i_auroc_vs_raw_e_pp", "full_p_auroc", "shield_p_auroc",
            "full_p_aupro", "shield_p_aupro", "num_images", "num_shield_wins",
            "num_full_memory_applied", "num_full_memory_blocked",
            "num_normal_memory_blocked", "num_anomaly_memory_blocked",
        ),
    )
    _write_csv(
        output_dir / "source_manifest.csv",
        sources.rows(),
        ("path", "size_bytes", "sha256", "scenarios", "roles"),
    )
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenarios": list(scenarios),
        "six_scenario_equal_weight_macro": tuple(scenarios) == SCENARIOS,
        "protocol": EXPECTED_PROTOCOL,
        "reproduction_gate": {
            "status": "GREEN" if gate_passed else "RED",
            "passed": gate_passed,
            "default_tolerance": tolerance,
            "p_aupro_tolerance": aupro_tolerance,
            "p_aupro_tolerance_reason": "PRO discrete numerical integration variation",
            "num_comparisons": len(reproduction_rows),
            "num_failed": len(failed_gate_rows),
            "max_abs_diff": max_gate_diff,
            "optimization_conclusion_allowed": gate_passed and tuple(scenarios) == SCENARIOS,
        },
        "macro_metrics": {
            variant: {metric: macro_lookup[(variant, metric)] for metric in METRICS}
            for variant in VARIANT_ORDER
        },
        "optimization_assessment": assessment,
        "routing_totals": {
            "num_test_images": sum(row["num_test_images"] for row in routing_rows),
            "num_shield_wins": sum(row["num_shield_wins"] for row in routing_rows),
            "num_full_memory_applied": sum(row["num_full_memory_applied"] for row in routing_rows),
            "num_full_memory_blocked": sum(row["num_full_memory_blocked"] for row in routing_rows),
            "num_normal_memory_blocked": sum(row["num_normal_memory_blocked"] for row in routing_rows),
            "num_anomaly_memory_blocked": sum(row["num_anomaly_memory_blocked"] for row in routing_rows),
        },
        "focus_classes": sorted(FOCUS_CLASSES),
        "output_files": [
            "reproduction_gate.csv",
            "scenario_metrics.csv",
            "macro_metrics.csv",
            "per_class_deltas.csv",
            "routing_by_scenario.csv",
            "routing_by_class.csv",
            "routing_transitions_resolved.csv",
            "focus_capsule_pcb1_fryum.csv",
            "source_manifest.csv",
            (
                "{}_audit_zh.md".format(scenarios[0])
                if len(scenarios) == 1 else "ROUTING_SHIELD_RESULT_AUDIT_ZH.md"
            ),
        ],
    }
    _write_json(output_dir / "routing_shield_audit_summary.json", result)
    report_name = (
        "{}_audit_zh.md".format(scenarios[0])
        if len(scenarios) == 1 else "ROUTING_SHIELD_RESULT_AUDIT_ZH.md"
    )
    _write_text(
        output_dir / report_name,
        _render_report(
            gate_passed,
            tolerance,
            macro_lookup,
            routing_rows,
            focus_rows,
            assessment,
            failed_gate_rows,
            scenarios,
            transition_rows,
            per_class_delta_rows,
            aupro_tolerance,
        ),
    )
    return result


def _validate_tri_state_protocol(
    summary: Mapping[str, Any], provenance: Mapping[str, Any], scenario: str
) -> None:
    if summary.get("evaluator") != "dinomaly_tailguard_tri_state_eval":
        raise ValueError("{} has an unexpected tri-state evaluator".format(scenario))
    if summary.get("no_training") is not True or provenance.get("no_training") is not True:
        raise ValueError("{} tri-state evaluation is not inference-only".format(scenario))
    protocol = summary.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("{} tri-state summary has no protocol".format(scenario))
    for key, expected in EXPECTED_PROTOCOL.items():
        observed = protocol.get(key)
        if expected is None:
            if observed is not None:
                raise ValueError("{} tri-state {} must be null".format(scenario, key))
        elif _float(observed, "{} tri-state protocol {}".format(scenario, key)) != expected:
            raise ValueError("{} tri-state protocol differs for {}".format(scenario, key))
    contract = provenance.get("single_pass_contract", {})
    for key in (
        "shared_full_checkpoint", "shared_retained_train_set", "one_train_encoder_pass",
        "one_test_joint_model_pass",
    ):
        if contract.get(key) is not True:
            raise ValueError("{} tri-state provenance violates {}".format(scenario, key))
    equivalence = contract.get("equivalence_audit", {})
    for key in ("max_abs_reconstruction_map_error", "max_abs_patch_feature_error"):
        if _float(equivalence.get(key), "{} tri-state {}".format(scenario, key)) > 1e-5:
            raise ValueError("{} tri-state forward equivalence failed".format(scenario))
    registry = provenance.get("tri_state_registry", {}).get("summary", {})
    policy = registry.get("policy", {})
    integrity = registry.get("integrity", {})
    if registry.get("runtime_decision_is_train_only") is not True:
        raise ValueError("{} tri-state decision is not declared train-only".format(scenario))
    if registry.get("oracle_columns_consumed") not in ([], None):
        raise ValueError("{} tri-state consumed oracle columns".format(scenario))
    if policy.get("runtime_uses_labels") is not False or policy.get("low_margin_threshold") is not None:
        raise ValueError("{} tri-state policy is not frozen disagreement-only".format(scenario))
    for key in (
        "sample_roles_cover_retained_once", "every_sample_has_a_route",
        "memory_members_are_unique", "shield_has_no_memory",
        "structural_and_memory_roles_are_separate",
    ):
        if integrity.get(key) is not True:
            raise ValueError("{} tri-state integrity failed: {}".format(scenario, key))
    memory_audit = summary.get("routing", {}).get("memory_ownership_audit", {})
    if _int(memory_audit.get("shield_banks"), scenario + " shield banks") != 0:
        raise ValueError("{} tri-state shield owns a patch bank".format(scenario))
    if _int(memory_audit.get("head_banks"), scenario + " head banks") != 0:
        raise ValueError("{} tri-state head owns a patch bank".format(scenario))


def _render_tri_state_report(
    scenarios: Sequence[str],
    gate_passed: bool,
    tolerance: float,
    aupro_tolerance: float,
    macro_lookup: Mapping[Tuple[str, str], float],
    assessment: Optional[Mapping[str, Any]],
    routing_rows: Sequence[Mapping[str, Any]],
    changed_rows: Sequence[Mapping[str, Any]],
    score_equivalence_rows: Sequence[Mapping[str, Any]],
) -> str:
    complete = tuple(scenarios) == SCENARIOS
    title = "六场景" if complete else "阶段性（{} 场景）".format(len(scenarios))
    lines = ["# Tri-state P {}结果审计".format(title), ""]
    if gate_passed:
        lines.append("**复现闸门：GREEN / PASSED。**")
    else:
        lines.append("**复现闸门：RED / BLOCKED。不得形成优化结论。**")
    lines.extend([
        "",
        "一般指标容差 `{}`；P-AUPRO 因 PRO 离散数值积分使用 `{}`。".format(
            tolerance, aupro_tolerance
        ),
        "",
        "## {}等权平均".format(title),
        "",
        "| 变体 | I-AUROC | P-AUROC | P-AUPRO |",
        "|---|---:|---:|---:|",
    ])
    labels = {
        "core": "Unified Reconstruction Core",
        "h_only": "+ H",
        "h_raw_e": "+ H + Raw-E",
        "full_v5": "原 Full",
        "full_retest": "联合重测 Full",
        "full_shield": "Full + Routing Shield",
        "tri_state": "Full + Tri-state P",
    }
    for variant in TRI_VARIANT_ORDER:
        lines.append("| {} | {} | {} | {} |".format(
            labels[variant],
            _markdown_percent(macro_lookup[(variant, "I-AUROC")]),
            _markdown_percent(macro_lookup[(variant, "P-AUROC")]),
            _markdown_percent(macro_lookup[(variant, "P-AUPRO")]),
        ))
    lines.append("")
    if complete and gate_passed and assessment is not None:
        lines.extend([
            "## 判定",
            "",
            "结论：**{}**。".format(assessment["verdict"]),
            "",
            "- Tri-state 相对原 Full：I-AUROC {:+.3f}、P-AUROC {:+.3f}、P-AUPRO {:+.3f} 个百分点。".format(
                100 * assessment["tri_state_minus_full"]["I-AUROC"],
                100 * assessment["tri_state_minus_full"]["P-AUROC"],
                100 * assessment["tri_state_minus_full"]["P-AUPRO"],
            ),
            "- Tri-state 相对 Raw-E：I-AUROC {:+.3f}、P-AUROC {:+.3f}、P-AUPRO {:+.3f} 个百分点。".format(
                100 * assessment["tri_state_minus_h_raw_e"]["I-AUROC"],
                100 * assessment["tri_state_minus_h_raw_e"]["P-AUROC"],
                100 * assessment["tri_state_minus_h_raw_e"]["P-AUPRO"],
            ),
            "",
        ])
    lines.extend(["## 路由规模", ""])
    lines.append(
        "当前 {} 个场景共 {} 张测试图；Tri-state memory 应用 {} 次；训练侧辅助尾样本 {} 个。".format(
            len(scenarios),
            sum(int(row["num_test_images"]) for row in routing_rows),
            sum(int(row["num_tri_state_memory_applied"]) for row in routing_rows),
            sum(int(row["num_auxiliary_memory_samples"]) for row in routing_rows),
        )
    )
    lines.append("")
    if changed_rows:
        lines.extend([
            "## 相对 Shield 的主指标变化",
            "",
            "| 场景 | 类别 | 指标 | Tri-state - Shield（百分点） |",
            "|---|---|---|---:|",
        ])
        for row in changed_rows:
            lines.append("| {scenario} | {class_name} | {metric} | {delta_pp:+.3f} |".format(
                **row
            ))
        lines.append("")
    if score_equivalence_rows:
        lines.extend([
            "## Auxiliary-tail 与 Raw-E 逐图等价性",
            "",
            "| 场景 | 类别 | 查询数 | final 最大差 | memory 最大差 |",
            "|---|---|---:|---:|---:|",
        ])
        for row in score_equivalence_rows:
            lines.append(
                "| {scenario} | {class_name} | {num_queries} | "
                "{max_abs_final_score_diff_vs_raw_e:.3e} | "
                "{max_abs_memory_score_diff_vs_raw_e:.3e} |".format(**row)
            )
        lines.extend([
            "",
            "上述差值若处于浮点误差量级，说明 tri-state 恢复的是 Raw-E 专属 memory 行为，而非仅回退 reconstruction。",
            "",
        ])
    lines.extend(["完整审计证据见同目录 CSV/JSON。", ""])
    return "\n".join(lines)


def audit_tri_state(
    results_root: Path,
    tri_state_root: Path,
    output_dir: Path,
    tolerance: float,
    scenarios: Sequence[str] = SCENARIOS,
    aupro_tolerance: float = 5e-6,
) -> Dict[str, Any]:
    scenarios = tuple(scenarios)
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("tri-state scenarios must be non-empty and unique")
    unknown = sorted(set(scenarios).difference(SCENARIOS))
    if unknown:
        raise ValueError("Unknown tri-state scenarios: {}".format(unknown))
    missing = missing_tri_state_inputs(results_root, tri_state_root, scenarios)
    if missing:
        raise FileNotFoundError(
            "{} tri-state inputs are missing; first: {}".format(len(missing), missing[0]["path"])
        )
    sources = SourceManifest()
    scenario_rows: List[Dict[str, Any]] = []
    gate_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []
    routing_rows: List[Dict[str, Any]] = []
    score_equivalence_rows: List[Dict[str, Any]] = []
    paired_causal_rows: List[Dict[str, Any]] = []

    for scenario in scenarios:
        old = {}
        for variant in ("core", "h_only"):
            path = results_root / (scenario + "_" + variant) / "tailguard" / "tailguard_summary.json"
            old[variant] = _read_json(sources.add(path, scenario, variant + "_summary"))
        for variant in ("h_raw_e", "full"):
            path = results_root / (scenario + "_" + variant) / "tailguard" / "memory" / "memory_eval_summary.json"
            old[variant] = _read_json(sources.add(path, scenario, variant + "_memory_eval"))
        directory = tri_state_root / scenario
        summary = _read_json(sources.add(
            directory / "tri_state_eval_summary.json", scenario, "tri_state_summary"
        ))
        provenance = _read_json(sources.add(
            directory / "tri_state_eval_provenance.json", scenario, "tri_state_provenance"
        ))
        per_class_csv = _read_csv(sources.add(
            directory / "tri_state_per_class_metrics.csv", scenario, "tri_state_per_class"
        ))
        scores = _read_csv(sources.add(
            directory / "tri_state_scores.csv", scenario, "tri_state_scores"
        ))
        sources.add(directory / "tri_state_route_audit.csv", scenario, "tri_state_routes")
        _validate_tri_state_protocol(summary, provenance, scenario)
        expected_checkpoint = (results_root / (scenario + "_full") / "final_model.pt").resolve()
        if Path(str(provenance.get("checkpoint", {}).get("path", ""))).resolve() != expected_checkpoint:
            raise ValueError("{} tri-state checkpoint does not match v5 Full".format(scenario))

        metrics = summary.get("metrics", {})
        if set(metrics) != {"recon", "full", "full_shield", "tri_state"}:
            raise ValueError("{} tri-state modes are incomplete".format(scenario))
        by_variant = {
            "core": _metric_values(old["core"]["final_eval_summary"], scenario + " core"),
            "h_only": _metric_values(old["h_only"]["final_eval_summary"], scenario + " h_only"),
            "h_raw_e": _metric_values(old["h_raw_e"]["summary"], scenario + " raw"),
            "full_v5": _metric_values(old["full"]["summary"], scenario + " full v5"),
            "full_retest": _metric_values(metrics["full"], scenario + " full retest"),
            "full_shield": _metric_values(metrics["full_shield"], scenario + " shield"),
            "tri_state": _metric_values(metrics["tri_state"], scenario + " tri-state"),
        }
        for variant, values in by_variant.items():
            scenario_rows.append({"scenario": scenario, "variant": variant, **values})
        for intervention in ("full_shield", "tri_state"):
            for metric in METRICS:
                paired_causal_rows.append({
                    "scenario": scenario,
                    "intervention": intervention,
                    "scope": "summary",
                    "class_id": "",
                    "class_name": "",
                    "metric": metric,
                    "paired_delta_vs_full": (
                        by_variant[intervention][metric] - by_variant["full_retest"][metric]
                    ),
                })

        old_modes = old["full"].get("metrics_by_mode", {})
        old_summary_modes = {
            "recon": _metric_values(old_modes["recon"]["summary"], scenario + " old recon"),
            "full": by_variant["full_v5"],
        }
        for mode in ("recon", "full"):
            current = _metric_values(metrics[mode], scenario + " new " + mode)
            for metric in METRICS:
                _add_gate_comparison(
                    gate_rows, scenario, "summary", mode, metric,
                    old_summary_modes[mode][metric], current[metric],
                    _gate_tolerance(metric, tolerance, aupro_tolerance),
                )

        mode_rows: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in per_class_csv:
            mode_rows[str(row.get("mode", ""))].append(row)
        if set(mode_rows) != {"recon", "full", "full_shield", "tri_state"}:
            raise ValueError("{} tri-state per-class modes are incomplete".format(scenario))
        indexed = {
            mode: _index_per_class(rows, scenario + " tri-state " + mode)
            for mode, rows in mode_rows.items()
        }
        old_class = {
            "recon": _index_per_class(
                old_modes["recon"]["per_class_metrics"], scenario + " old recon"
            ),
            "full": _index_per_class(old["full"]["per_class_metrics"], scenario + " old full"),
        }
        keys = set(indexed["full"])
        if any(set(frame) != keys for frame in list(indexed.values()) + list(old_class.values())):
            raise ValueError("{} tri-state class identities differ".format(scenario))
        raw_class = _index_per_class(
            old["h_raw_e"]["per_class_metrics"], scenario + " raw per-class"
        )
        if set(raw_class) != keys:
            raise ValueError("{} raw per-class identities differ".format(scenario))
        for mode in ("recon", "full"):
            for class_id, class_name in sorted(keys):
                for metric in REPRODUCTION_PER_CLASS_METRICS:
                    _add_gate_comparison(
                        gate_rows, scenario, "per_class", mode, metric,
                        old_class[mode][(class_id, class_name)][metric],
                        indexed[mode][(class_id, class_name)][metric],
                        _gate_tolerance(metric, tolerance, aupro_tolerance),
                        class_id, class_name,
                    )
        for class_id, class_name in sorted(keys):
            for metric in METRICS:
                tri = indexed["tri_state"][(class_id, class_name)][metric]
                shield = indexed["full_shield"][(class_id, class_name)][metric]
                full = indexed["full"][(class_id, class_name)][metric]
                raw = raw_class[(class_id, class_name)][metric]
                delta_rows.append({
                    "scenario": scenario,
                    "class_id": class_id,
                    "class_name": class_name,
                    "metric": metric,
                    "h_raw_e": raw,
                    "full": full,
                    "full_shield": shield,
                    "tri_state": tri,
                    "tri_state_minus_full": tri - full,
                    "tri_state_minus_shield": tri - shield,
                    "tri_state_minus_h_raw_e": tri - raw,
                })
                for intervention in ("full_shield", "tri_state"):
                    paired_causal_rows.append({
                        "scenario": scenario,
                        "intervention": intervention,
                        "scope": "per_class",
                        "class_id": class_id,
                        "class_name": class_name,
                        "metric": metric,
                        "paired_delta_vs_full": (
                            indexed[intervention][(class_id, class_name)][metric] - full
                        ),
                    })
        route = summary.get("routing", {})
        routing_rows.append({
            "scenario": scenario,
            "num_test_images": _int(route.get("num_test_images"), scenario + " test images"),
            "num_full_shields": _int(route.get("num_full_shields"), scenario + " full shields"),
            "num_tri_state_routes": _int(route.get("num_tri_state_routes"), scenario + " routes"),
            "num_tri_state_shields": _int(route.get("num_tri_state_shields"), scenario + " tri shields"),
            "num_auxiliary_tail_classes": _int(route.get("num_auxiliary_tail_classes"), scenario + " aux classes"),
            "num_auxiliary_memory_samples": _int(route.get("num_auxiliary_memory_samples"), scenario + " aux samples"),
            "num_tri_state_memory_applied": _int(route.get("num_tri_state_memory_applied"), scenario + " applied"),
            "scores_num_rows": len(scores),
            "scores_num_memory_applied": sum(
                _int(row.get("tri_state_mem_applied"), scenario + " score applied") for row in scores
            ),
            "scores_num_shield_blocked": sum(
                _int(row.get("full_memory_blocked_by_shield"), scenario + " shield blocked")
                for row in scores
            ),
        })
        if routing_rows[-1]["scores_num_rows"] != routing_rows[-1]["num_test_images"]:
            raise ValueError("{} tri-state score row count differs".format(scenario))
        if routing_rows[-1]["scores_num_memory_applied"] != routing_rows[-1]["num_tri_state_memory_applied"]:
            raise ValueError("{} tri-state applied count differs".format(scenario))

        # When tri-state introduces an auxiliary bank, verify whether it is an
        # exact recovery of the corresponding Raw-E bank at the score level.
        auxiliary_classes = sorted({
            str(row.get("class_name", ""))
            for row in scores
            if str(row.get("tri_state_route_type", "")) == "auxiliary_tail"
        })
        if auxiliary_classes:
            raw_scores_path = (
                results_root / (scenario + "_h_raw_e") / "tailguard" / "memory"
                / "memory_eval_scores.csv"
            )
            raw_scores = _read_csv(sources.add(
                raw_scores_path, scenario, "raw_e_memory_scores"
            ))

            def stable_test_key(row: Mapping[str, Any]) -> Tuple[str, str]:
                path = str(row.get("img_path", "")).replace("\\", "/")
                marker = "/test/"
                if marker not in path:
                    raise ValueError("{} score path has no /test/: {}".format(scenario, path))
                return str(row.get("class_name", "")), "test/" + path.split(marker, 1)[1]

            raw_by_key = {stable_test_key(row): row for row in raw_scores}
            if len(raw_by_key) != len(raw_scores):
                raise ValueError("{} Raw-E scores have duplicate test identities".format(scenario))
            for class_name in auxiliary_classes:
                tri_rows = [row for row in scores if row.get("class_name") == class_name]
                differences: Dict[str, List[float]] = {
                    "final_score": [], "memory_score": [], "recon_score": [],
                }
                for tri_row in tri_rows:
                    key = stable_test_key(tri_row)
                    if key not in raw_by_key:
                        raise ValueError("{} auxiliary query is absent from Raw-E".format(scenario))
                    raw_row = raw_by_key[key]
                    differences["final_score"].append(
                        _float(tri_row["tri_state_final_score"], scenario + " tri final")
                        - _float(raw_row["final_score"], scenario + " raw final")
                    )
                    differences["memory_score"].append(
                        _float(tri_row["tri_state_memory_score"], scenario + " tri memory")
                        - _float(raw_row["memory_score"], scenario + " raw memory")
                    )
                    differences["recon_score"].append(
                        _float(tri_row["recon_score"], scenario + " tri recon")
                        - _float(raw_row["recon_score"], scenario + " raw recon")
                    )
                score_equivalence_rows.append({
                    "scenario": scenario,
                    "class_name": class_name,
                    "num_queries": len(tri_rows),
                    "max_abs_final_score_diff_vs_raw_e": max(
                        abs(value) for value in differences["final_score"]
                    ),
                    "max_abs_memory_score_diff_vs_raw_e": max(
                        abs(value) for value in differences["memory_score"]
                    ),
                    "max_abs_recon_score_diff_vs_raw_e": max(
                        abs(value) for value in differences["recon_score"]
                    ),
                })

    gate_passed = not any(not row["passed"] for row in gate_rows)
    macro_lookup: Dict[Tuple[str, str], float] = {}
    macro_rows = []
    for variant in TRI_VARIANT_ORDER:
        rows = [row for row in scenario_rows if row["variant"] == variant]
        if len(rows) != len(scenarios):
            raise RuntimeError("Tri-state variant {} is incomplete".format(variant))
        for metric in METRICS:
            value = _mean(float(row[metric]) for row in rows)
            macro_lookup[(variant, metric)] = value
            macro_rows.append({"variant": variant, "metric": metric, "macro_value": value})
    assessment = None
    if gate_passed and tuple(scenarios) == SCENARIOS:
        tri_minus_full = {
            metric: macro_lookup[("tri_state", metric)] - macro_lookup[("full_v5", metric)]
            for metric in METRICS
        }
        tri_minus_raw = {
            metric: macro_lookup[("tri_state", metric)] - macro_lookup[("h_raw_e", metric)]
            for metric in METRICS
        }
        checks = {
            "reaches_raw_e_i_auroc": tri_minus_raw["I-AUROC"] >= -tolerance,
            "preserves_full_p_auroc": tri_minus_full["P-AUROC"] >= -tolerance,
            "preserves_full_p_aupro": tri_minus_full["P-AUPRO"] >= -tolerance,
        }
        if all(checks.values()):
            verdict = "SUCCESS：Tri-state 同时达到图像与像素目标"
        elif tri_minus_full["I-AUROC"] > tolerance:
            verdict = "PARTIAL：Tri-state 改善 Full，但未满足全部目标"
        else:
            verdict = "NO-GAIN：Tri-state 未改善原 Full 图像宏平均"
        assessment = {
            "verdict": verdict,
            "checks": checks,
            "tri_state_minus_full": tri_minus_full,
            "tri_state_minus_h_raw_e": tri_minus_raw,
        }
    changed_main = [
        {
            "scenario": row["scenario"],
            "class_name": row["class_name"],
            "metric": row["metric"],
            "delta_pp": 100.0 * float(row["tri_state_minus_shield"]),
        }
        for row in delta_rows
        if row["metric"] in REPRODUCTION_PER_CLASS_METRICS
        and abs(float(row["tri_state_minus_shield"])) > 1e-12
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "tri_state_reproduction_gate.csv", gate_rows, (
        "scenario", "scope", "mode", "class_id", "class_name", "metric",
        "old_value", "new_value", "signed_diff", "abs_diff", "tolerance", "passed",
    ))
    _write_csv(output_dir / "tri_state_scenario_metrics.csv", scenario_rows, (
        "scenario", "variant", *METRICS,
    ))
    _write_csv(output_dir / "tri_state_macro_metrics.csv", macro_rows, (
        "variant", "metric", "macro_value",
    ))
    _write_csv(output_dir / "tri_state_per_class_deltas.csv", delta_rows, (
        "scenario", "class_id", "class_name", "metric", "h_raw_e", "full",
        "full_shield", "tri_state", "tri_state_minus_full", "tri_state_minus_shield",
        "tri_state_minus_h_raw_e",
    ))
    _write_csv(output_dir / "tri_state_routing_by_scenario.csv", routing_rows, (
        "scenario", "num_test_images", "num_full_shields", "num_tri_state_routes",
        "num_tri_state_shields", "num_auxiliary_tail_classes",
        "num_auxiliary_memory_samples", "num_tri_state_memory_applied", "scores_num_rows",
        "scores_num_memory_applied", "scores_num_shield_blocked",
    ))
    _write_csv(output_dir / "tri_state_source_manifest.csv", sources.rows(), (
        "path", "size_bytes", "sha256", "scenarios", "roles",
    ))
    _write_csv(output_dir / "tri_state_auxiliary_raw_e_score_equivalence.csv", score_equivalence_rows, (
        "scenario", "class_name", "num_queries", "max_abs_final_score_diff_vs_raw_e",
        "max_abs_memory_score_diff_vs_raw_e", "max_abs_recon_score_diff_vs_raw_e",
    ))
    _write_csv(output_dir / "tri_state_paired_causal_deltas.csv", paired_causal_rows, (
        "scenario", "intervention", "scope", "class_id", "class_name", "metric",
        "paired_delta_vs_full",
    ))
    paired_causal_summary = {}
    for intervention in ("full_shield", "tri_state"):
        rows = [row for row in paired_causal_rows if row["intervention"] == intervention]
        paired_causal_summary[intervention] = {
            "max_abs_delta_vs_full": max(abs(float(row["paired_delta_vs_full"])) for row in rows),
            "all_metrics_exactly_equal_to_full": all(
                float(row["paired_delta_vs_full"]) == 0.0 for row in rows
            ),
        }
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenarios": list(scenarios),
        "six_scenario_equal_weight_macro": tuple(scenarios) == SCENARIOS,
        "reproduction_gate": {
            "status": "GREEN" if gate_passed else "RED",
            "passed": gate_passed,
            "default_tolerance": tolerance,
            "p_aupro_tolerance": aupro_tolerance,
            "num_comparisons": len(gate_rows),
            "num_failed": sum(not row["passed"] for row in gate_rows),
            "max_abs_diff": max(float(row["abs_diff"]) for row in gate_rows),
            "optimization_conclusion_allowed": gate_passed and tuple(scenarios) == SCENARIOS,
        },
        "macro_metrics": {
            variant: {metric: macro_lookup[(variant, metric)] for metric in METRICS}
            for variant in TRI_VARIANT_ORDER
        },
        "optimization_assessment": assessment,
        "routing_totals": {
            "num_test_images": sum(row["num_test_images"] for row in routing_rows),
            "num_tri_state_memory_applied": sum(
                row["num_tri_state_memory_applied"] for row in routing_rows
            ),
            "num_auxiliary_memory_samples": sum(
                row["num_auxiliary_memory_samples"] for row in routing_rows
            ),
            "num_shield_blocked": sum(row["scores_num_shield_blocked"] for row in routing_rows),
        },
        "auxiliary_raw_e_score_equivalence": score_equivalence_rows,
        "paired_causal_audit": paired_causal_summary,
    }
    _write_json(output_dir / "tri_state_audit_summary.json", result)
    report_name = (
        "{}_tri_state_audit_zh.md".format(scenarios[0])
        if len(scenarios) == 1 else "TRI_STATE_RESULT_AUDIT_ZH.md"
    )
    _write_text(output_dir / report_name, _render_tri_state_report(
        scenarios, gate_passed, tolerance, aupro_tolerance, macro_lookup,
        assessment, routing_rows, changed_main, score_equivalence_rows,
    ))
    return result


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Strictly audit and aggregate six Routing Shield evaluations"
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=repo_root.parent / "results" / "tailguard" / "dependency_ablation_v5",
    )
    parser.add_argument(
        "--shield-root",
        type=Path,
        default=repo_root / "analysis_outputs" / "e_routing_fix_v1" / "shield",
    )
    parser.add_argument(
        "--tri-state-root",
        type=Path,
        default=repo_root / "analysis_outputs" / "e_routing_fix_v1" / "tri_state_eval",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "analysis_outputs" / "e_routing_fix_v1" / "audit",
    )
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--aupro-tolerance", type=float, default=5e-6)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIOS,
        default=list(SCENARIOS),
        help="Audit a subset while evaluations are still running",
    )
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Only report missing inputs; do not create audit outputs",
    )
    parser.add_argument(
        "--audit-kind",
        choices=("shield", "tri_state"),
        default="shield",
        help="Select the independent result family to audit",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.audit_kind == "tri_state":
        missing = missing_tri_state_inputs(
            args.results_root, args.tri_state_root, args.scenarios
        )
    else:
        missing = missing_inputs(args.results_root, args.shield_root, args.scenarios)
    if args.inventory_only:
        print(json.dumps({
            "ready": not missing,
            "num_missing": len(missing),
            "missing": missing,
        }, ensure_ascii=False, indent=2))
        return 0 if not missing else 2
    if missing:
        print("Routing Shield audit is not ready: {} inputs are missing.".format(len(missing)), file=sys.stderr)
        for row in missing:
            print("- {scenario}: {path}".format(**row), file=sys.stderr)
        return 2
    try:
        if args.audit_kind == "tri_state":
            result = audit_tri_state(
                args.results_root,
                args.tri_state_root,
                args.output_dir,
                args.tolerance,
                args.scenarios,
                args.aupro_tolerance,
            )
        else:
            result = audit(
                args.results_root,
                args.shield_root,
                args.output_dir,
                args.tolerance,
                args.scenarios,
                args.aupro_tolerance,
            )
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print("Routing Shield audit failed: {}".format(error), file=sys.stderr)
        return 4
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["reproduction_gate"]["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
