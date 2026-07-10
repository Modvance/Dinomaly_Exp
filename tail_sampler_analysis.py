import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from tail_sampler import build_tail_sampler


def _safe_binary_metric(metric_fn, y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    mask = ~pd.isna(y_true) & ~pd.isna(y_score)
    mask = mask & np.isin(y_true, [0, 1])
    y_true = y_true[mask]
    y_score = y_score[mask]
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float('nan')
    return float(metric_fn(y_true.astype(int), y_score.astype(float)))


def build_tail_ground_truth(metadata_df: pd.DataFrame, args) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], int, str]:
    metadata_df = metadata_df.copy()
    class_counts = metadata_df.groupby('class_id').size().to_dict()
    gt_class_counts = metadata_df['class_id'].map(class_counts).astype(int).to_numpy()
    gt_mode = getattr(args, 'tailsampler_gt_mode', 'true_class')

    if gt_mode == 'true_class':
        threshold = int(getattr(args, 'tailsampler_tail_count_thr', 8))
        is_gt_tail = (gt_class_counts <= threshold).astype(int)
        gt_rule = f'class_count<={threshold}'
        return gt_class_counts, is_gt_tail, {int(k): int(v) for k, v in class_counts.items()}, threshold, gt_rule

    if gt_mode == 'dataset_rule':
        dataset_name = str(getattr(args, 'tailsampler_dataset_name', '') or '')
        dataset_name_lower = dataset_name.lower()
        if 'step_k1' in dataset_name_lower or 'step-k1' in dataset_name_lower:
            threshold = 1
            gt_rule = 'dataset_rule:step_k1'
        elif 'step_k4' in dataset_name_lower or 'step-k4' in dataset_name_lower:
            threshold = 4
            gt_rule = 'dataset_rule:step_k4'
        elif 'pareto' in dataset_name_lower:
            threshold = 19
            gt_rule = 'dataset_rule:pareto(<20)'
        else:
            raise ValueError('tailsampler_gt_mode=dataset_rule requires dataset name containing step_k1, step_k4, or pareto')
        is_gt_tail = (gt_class_counts <= threshold).astype(int) if threshold != 19 else (gt_class_counts < 20).astype(int)
        return gt_class_counts, is_gt_tail, {int(k): int(v) for k, v in class_counts.items()}, threshold, gt_rule

    raise ValueError(f'unsupported tailsampler_gt_mode: {gt_mode}')


def compute_selection_metrics(selected: np.ndarray, is_gt_tail: np.ndarray) -> Dict[str, float]:
    selected = np.asarray(selected).astype(int)
    is_gt_tail = np.asarray(is_gt_tail).astype(int)
    tp = int(np.sum((selected == 1) & (is_gt_tail == 1)))
    fp = int(np.sum((selected == 1) & (is_gt_tail == 0)))
    tn = int(np.sum((selected == 0) & (is_gt_tail == 0)))
    fn = int(np.sum((selected == 0) & (is_gt_tail == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'selected_ratio': float(selected.mean()) if len(selected) > 0 else 0.0,
    }


def compute_ranking_metrics(tail_score: np.ndarray, is_gt_tail: np.ndarray) -> Dict[str, float]:
    return {
        'auroc': _safe_binary_metric(roc_auc_score, is_gt_tail, tail_score),
        'auprc': _safe_binary_metric(average_precision_score, is_gt_tail, tail_score),
    }


def run_tail_sampler_analysis(embeddings, metadata_df: pd.DataFrame, args):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError('embeddings must have shape [N, D]')
    if len(metadata_df) != embeddings.shape[0]:
        raise ValueError('metadata and embeddings size mismatch')

    sampler, sampler_name = build_tail_sampler(
        sampler_type=getattr(args, 'tailsampler_type', 'adaptive_trim_mode'),
        threshold_type=getattr(args, 'tailsampler_th_type', None),
        vote_type=getattr(args, 'tailsampler_vote_type', None),
        percentile=float(getattr(args, 'tailsampler_percentile', 0.15)),
    )

    feature_tensor = torch.from_numpy(embeddings)
    _, tail_indices, class_sizes_pred = sampler.run(feature_tensor, return_class_sizes=True)
    class_sizes_pred = class_sizes_pred.detach().cpu().numpy().astype(float)

    selected = np.zeros(len(metadata_df), dtype=int)
    selected[tail_indices.detach().cpu().numpy()] = 1

    gt_class_counts, is_gt_tail, class_counts, tail_max_count, gt_rule = build_tail_ground_truth(metadata_df, args)
    tail_score = -class_sizes_pred
    ranking_metrics = compute_ranking_metrics(tail_score, is_gt_tail)
    selection_metrics = compute_selection_metrics(selected, is_gt_tail)

    details_df = metadata_df.copy()
    details_df['gt_class_count'] = gt_class_counts.astype(int)
    details_df['is_gt_tail'] = is_gt_tail.astype(int)
    details_df['pred_class_size'] = class_sizes_pred.astype(float)
    details_df['tail_score'] = tail_score.astype(float)
    details_df['is_selected'] = selected.astype(int)

    summary_row = {
        'run_name': getattr(args, 'save_name', 'tailsampler_run'),
        'dataset_name': getattr(args, 'tailsampler_dataset_name', os.path.basename(str(getattr(args, 'data_path', '')))),
        'sampler_name': sampler_name,
        'tailsampler_type': getattr(args, 'tailsampler_type', 'adaptive_trim_mode'),
        'tail_th_type': getattr(args, 'tailsampler_th_type', None),
        'vote_type': getattr(args, 'tailsampler_vote_type', None),
        'embedding_source': getattr(args, 'tailsampler_embedding_source', 'encoder'),
        'num_samples': int(len(details_df)),
        'num_classes': int(len(class_counts)),
        'gt_mode': getattr(args, 'tailsampler_gt_mode', 'true_class'),
        'gt_rule': gt_rule,
        'tail_max_count': int(tail_max_count),
        'num_gt_tail_samples': int(is_gt_tail.sum()),
        'num_selected': int(selected.sum()),
        'selection_precision': float(selection_metrics['precision']),
        'selection_recall': float(selection_metrics['recall']),
        'selection_f1': float(selection_metrics['f1']),
        'selection_tp': int(selection_metrics['tp']),
        'selection_fp': int(selection_metrics['fp']),
        'selection_tn': int(selection_metrics['tn']),
        'selection_fn': int(selection_metrics['fn']),
        'selected_ratio': float(selection_metrics['selected_ratio']),
        'tail_auroc': float(ranking_metrics['auroc']) if not np.isnan(ranking_metrics['auroc']) else None,
        'tail_auprc': float(ranking_metrics['auprc']) if not np.isnan(ranking_metrics['auprc']) else None,
        'mean_pred_class_size': float(np.mean(class_sizes_pred)) if len(class_sizes_pred) > 0 else None,
        'mean_pred_class_size_tail': float(np.mean(class_sizes_pred[is_gt_tail == 1])) if np.any(is_gt_tail == 1) else None,
        'mean_pred_class_size_head': float(np.mean(class_sizes_pred[is_gt_tail == 0])) if np.any(is_gt_tail == 0) else None,
    }

    return summary_row, details_df


def save_tail_sampler_artifacts(output_dir: str, summary_row: Dict, details_df: pd.DataFrame, metadata: Dict, save_details: bool = False):
    os.makedirs(output_dir, exist_ok=True)

    summary_df = pd.DataFrame([summary_row])
    summary_csv_path = os.path.join(output_dir, 'sampler_analysis.csv')
    summary_df.to_csv(summary_csv_path, index=False)

    aggregate_csv_path = os.path.join(output_dir, 'sampler_analysis_aggregate.csv')
    summary_df.to_csv(aggregate_csv_path, index=False)

    details_csv_path = os.path.join(output_dir, 'sampler_analysis_details.csv')
    if save_details:
        details_df.to_csv(details_csv_path, index=False)

    summary_json_path = os.path.join(output_dir, 'sampler_analysis_summary.json')
    payload = {
        'summary': summary_row,
        'metadata': metadata,
        'artifacts': {
            'sampler_analysis_csv': summary_csv_path,
            'sampler_analysis_aggregate_csv': aggregate_csv_path,
            'sampler_analysis_details_csv': details_csv_path if save_details else None,
        },
    }
    with open(summary_json_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    return {
        'sampler_analysis_csv': summary_csv_path,
        'sampler_analysis_aggregate_csv': aggregate_csv_path,
        'sampler_analysis_details_csv': details_csv_path if save_details else None,
        'sampler_analysis_summary_json': summary_json_path,
    }
