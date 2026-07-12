import json
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch


def _make_json_safe(value):
    if isinstance(value, dict):
        return {key: _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _update_summary_json(summary_path: str, extra_summary: Optional[Dict] = None):
    payload = {}
    if os.path.isfile(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as file:
            payload = json.load(file)
    if extra_summary is not None:
        payload.update(_make_json_safe(extra_summary))
    with open(summary_path, 'w', encoding='utf-8') as file:
        json.dump(_make_json_safe(payload), file, indent=2, ensure_ascii=False)
    return payload


def save_tailguard_prepare_artifacts(output_dir: str,
                                     candidates_df: pd.DataFrame,
                                     group_assignments_df: pd.DataFrame,
                                     train_metadata_df: pd.DataFrame,
                                     metadata: Dict,
                                     cls_embeddings_payload: Optional[Dict] = None,
                                     analysis_details_df: Optional[pd.DataFrame] = None):
    os.makedirs(output_dir, exist_ok=True)

    candidates_path = os.path.join(output_dir, 'tailguard_candidates.csv')
    group_assignments_path = os.path.join(output_dir, 'group_assignments.csv')
    train_metadata_path = os.path.join(output_dir, 'tailguard_train_metadata.csv')
    summary_path = os.path.join(output_dir, 'tailguard_prepare_summary.json')

    candidates_df.to_csv(candidates_path, index=False)
    group_assignments_df.to_csv(group_assignments_path, index=False)
    train_metadata_df.to_csv(train_metadata_path, index=False)

    cls_embeddings_path = None
    if cls_embeddings_payload is not None:
        cls_embeddings_path = os.path.join(output_dir, 'tailguard_cls_embeddings.pt')
        torch.save(cls_embeddings_payload, cls_embeddings_path)

    analysis_details_path = None
    if analysis_details_df is not None:
        analysis_details_path = os.path.join(output_dir, 'tail_sampler_analysis_details.csv')
        analysis_details_df.to_csv(analysis_details_path, index=False)

    payload = {
        'metadata': _make_json_safe(metadata),
        'artifacts': {
            'tailguard_candidates_csv': candidates_path,
            'group_assignments_csv': group_assignments_path,
            'tailguard_train_metadata_csv': train_metadata_path,
            'tailguard_cls_embeddings_pt': cls_embeddings_path,
            'tail_sampler_analysis_details_csv': analysis_details_path,
        },
    }
    with open(summary_path, 'w', encoding='utf-8') as file:
        json.dump(_make_json_safe(payload), file, indent=2, ensure_ascii=False)

    return {
        'tailguard_candidates_csv': candidates_path,
        'group_assignments_csv': group_assignments_path,
        'tailguard_train_metadata_csv': train_metadata_path,
        'tailguard_cls_embeddings_pt': cls_embeddings_path,
        'tail_sampler_analysis_details_csv': analysis_details_path,
        'tailguard_prepare_summary_json': summary_path,
    }


def save_tailguard_iteration_artifacts(iter_dir: str,
                                       global_group_metrics_df: pd.DataFrame,
                                       sample_tailguard_scores_df: pd.DataFrame,
                                       extra_summary: Optional[Dict] = None):
    os.makedirs(iter_dir, exist_ok=True)
    global_metrics_path = os.path.join(iter_dir, 'global_group_metrics.csv')
    sample_scores_path = os.path.join(iter_dir, 'sample_tailguard_scores.csv')
    summary_path = os.path.join(iter_dir, 'summary.json')

    global_group_metrics_df.to_csv(global_metrics_path, index=False)
    sample_tailguard_scores_df.to_csv(sample_scores_path, index=False)
    _update_summary_json(summary_path, extra_summary=extra_summary)

    return {
        'global_group_metrics_csv': global_metrics_path,
        'sample_tailguard_scores_csv': sample_scores_path,
        'summary_json': summary_path,
    }


def save_tailguard_prune_artifacts(iter_dir: str, delete_plan: Dict, extra_summary: Optional[Dict] = None):
    os.makedirs(iter_dir, exist_ok=True)

    tail_support_path = os.path.join(iter_dir, 'tail_support_samples.csv')
    tail_suspicious_path = os.path.join(iter_dir, 'tail_suspicious_samples.csv')
    risk_table_path = os.path.join(iter_dir, 'tailguard_risk_table.csv')
    kept_path = os.path.join(iter_dir, 'tailguard_kept_samples.csv')
    removed_path = os.path.join(iter_dir, 'tailguard_removed_samples.csv')
    summary_path = os.path.join(iter_dir, 'tailguard_prune_summary.json')

    if len(delete_plan['tail_support_samples']) > 0:
        delete_plan['tail_support_samples'].to_csv(tail_support_path, index=False)
    else:
        pd.DataFrame(columns=list(delete_plan['tailguard_risk_table'].columns)).to_csv(tail_support_path, index=False)

    if len(delete_plan['tail_suspicious_samples']) > 0:
        delete_plan['tail_suspicious_samples'].to_csv(tail_suspicious_path, index=False)
    else:
        pd.DataFrame(columns=list(delete_plan['tailguard_risk_table'].columns)).to_csv(tail_suspicious_path, index=False)

    delete_plan['tailguard_risk_table'].to_csv(risk_table_path, index=False)
    delete_plan['kept_samples'].to_csv(kept_path, index=False)
    delete_plan['pruned_samples'].to_csv(removed_path, index=False)

    payload = dict(delete_plan['summary'])
    if extra_summary is not None:
        payload.update(extra_summary)
    with open(summary_path, 'w', encoding='utf-8') as file:
        json.dump(_make_json_safe(payload), file, indent=2, ensure_ascii=False)

    return {
        'tail_support_samples_csv': tail_support_path,
        'tail_suspicious_samples_csv': tail_suspicious_path,
        'tailguard_risk_table_csv': risk_table_path,
        'tailguard_kept_samples_csv': kept_path,
        'tailguard_removed_samples_csv': removed_path,
        'tailguard_prune_summary_json': summary_path,
    }


def save_tailguard_memory_artifacts(output_dir: str,
                                    memory_bank: Dict,
                                    score_df: pd.DataFrame,
                                    per_class_metrics,
                                    summary: Dict,
                                    metadata: Dict,
                                    metrics_by_mode: Optional[Dict] = None):
    os.makedirs(output_dir, exist_ok=True)

    memory_bank_path = os.path.join(output_dir, 'tail_group_memory_bank.pt')
    bank_summary_path = os.path.join(output_dir, 'tail_group_memory_bank_summary.csv')
    score_df_path = os.path.join(output_dir, 'tail_group_memory_eval_scores.csv')
    summary_path = os.path.join(output_dir, 'tail_group_memory_eval_summary.json')

    torch.save(memory_bank, memory_bank_path)

    bank_rows = []
    for group_id, entry in sorted(memory_bank.items()):
        bank_rows.append({
            'group_id': int(group_id),
            'num_images': int(entry['num_images']),
            'num_patches': int(entry['num_patches']),
            'feature_dim': int(entry['feature_dim']),
            'spatial_h': int(entry['spatial_size'][0]),
            'spatial_w': int(entry['spatial_size'][1]),
            'adaptive_angle': float(entry['adaptive_angle']),
        })
    pd.DataFrame(bank_rows).to_csv(bank_summary_path, index=False)
    score_df.to_csv(score_df_path, index=False)

    payload = {
        'summary': summary,
        'per_class_metrics': per_class_metrics,
        'metadata': metadata,
    }
    if metrics_by_mode is not None:
        payload['metrics_by_mode'] = metrics_by_mode
    with open(summary_path, 'w', encoding='utf-8') as file:
        json.dump(_make_json_safe(payload), file, indent=2, ensure_ascii=False)

    return {
        'tail_group_memory_bank_pt': memory_bank_path,
        'tail_group_memory_bank_summary_csv': bank_summary_path,
        'tail_group_memory_eval_scores_csv': score_df_path,
        'tail_group_memory_eval_summary_json': summary_path,
    }


def save_tailguard_summary(output_dir: str, summary: Dict):
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, 'tailguard_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as file:
        json.dump(_make_json_safe(summary), file, indent=2, ensure_ascii=False)
    return summary_path
