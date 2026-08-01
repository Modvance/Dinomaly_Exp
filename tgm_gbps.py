"""GBPS helpers for the deprecated TGM method."""

import os
import re
from typing import Dict, Optional

import numpy as np
import pandas as pd

from gbps_runner_utils import build_gbps_hard_delete_plan
from safe_gbps import (
    compute_gbps_u_with_bootstrap,
    compute_group_sep_conf,
    fit_group_gmms,
    robust_normalize_group_scores,
)


def _safe_float(value, default=None):
    if value is None:
        return default
    try:
        value = float(value)
    except Exception:
        return default
    if np.isnan(value) or np.isinf(value):
        return default
    return value


def _build_tail_metadata_view(tgm_metadata_df: pd.DataFrame) -> pd.DataFrame:
    columns = ['sample_idx', 'sample_key', 'img_path', 'tail_candidate', 'pred_class_size', 'tail_score', 'group_id']
    metadata_df = tgm_metadata_df[columns].copy()
    metadata_df['sample_idx'] = metadata_df['sample_idx'].astype(int)
    metadata_df['sample_key'] = metadata_df['sample_key'].astype(int)
    metadata_df['group_id'] = metadata_df['group_id'].astype(int)
    metadata_df['tail_candidate'] = metadata_df['tail_candidate'].astype(int)
    return metadata_df


def compute_global_group_risk(sample_group_scores_df: pd.DataFrame, args) -> pd.DataFrame:
    if len(sample_group_scores_df) == 0:
        return pd.DataFrame(columns=[
            'group_id', 'group_size', 'group_score_median', 'global_z_score', 'global_p_high',
            'global_delta_bic', 'global_sep', 'global_conf', 'global_pi_high', 'global_is_active',
            'global_q', 'global_group_risk',
        ])

    per_group_df = sample_group_scores_df.groupby('group_id', sort=True).agg(
        group_size=('sample_idx', 'size'),
        group_score_median=('image_score', 'median'),
    ).reset_index()
    per_group_df['group_id'] = per_group_df['group_id'].astype(int)
    per_group_df['group_size'] = per_group_df['group_size'].astype(int)

    if not bool(getattr(args, 'tgm_enable_global_group_risk', False)) or len(per_group_df) == 1:
        per_group_df['global_z_score'] = 0.0
        per_group_df['global_p_high'] = 0.0
        per_group_df['global_delta_bic'] = 0.0
        per_group_df['global_sep'] = 0.0
        per_group_df['global_conf'] = 0.0
        per_group_df['global_pi_high'] = 0.0
        per_group_df['global_is_active'] = False
        per_group_df['global_q'] = 0.0
        per_group_df['global_group_risk'] = 0.0
        return per_group_df

    z_scores, _ = robust_normalize_group_scores(per_group_df['group_score_median'].to_numpy(dtype=float))
    gmm_stats = fit_group_gmms(z_scores, eps=1e-6, random_state=0)
    sep, conf = compute_group_sep_conf(gmm_stats)
    delta_bic = _safe_float(gmm_stats['delta_bic'], 0.0)
    pi_high = _safe_float(gmm_stats['pi_high'], 0.0)
    is_active = (
        delta_bic > float(getattr(args, 'tgm_global_tau_bic', 0.0))
        and float(sep) > float(getattr(args, 'tgm_global_tau_sep', 0.5))
        and float(conf) > float(getattr(args, 'tgm_global_tau_conf', 0.1))
        and float(getattr(args, 'tgm_global_pi_min', 0.05)) <= pi_high <= float(getattr(args, 'tgm_global_pi_max', 0.95))
    )
    q_global = 1.0 if is_active else 0.0
    posterior_high = np.asarray(gmm_stats['posterior_high'], dtype=float)

    per_group_df['global_z_score'] = z_scores.astype(float)
    per_group_df['global_p_high'] = posterior_high.astype(float)
    per_group_df['global_delta_bic'] = delta_bic
    per_group_df['global_sep'] = float(sep)
    per_group_df['global_conf'] = float(conf)
    per_group_df['global_pi_high'] = pi_high
    per_group_df['global_is_active'] = bool(is_active)
    per_group_df['global_q'] = float(q_global)
    per_group_df['global_group_risk'] = (float(q_global) * posterior_high).astype(float)
    return per_group_df


def run_tgm_gbps_iteration(scored_df: pd.DataFrame,
                                 group_assignments_df: pd.DataFrame,
                                 tgm_metadata_df: pd.DataFrame,
                                 args) -> Dict:
    gbps_result = compute_gbps_u_with_bootstrap(
        scored_df.copy(),
        group_assignments_df.copy(),
        B=int(getattr(args, 'gbps_bootstrap_B', 20)),
        args=args,
    )
    sample_group_scores_df = gbps_result['sample_group_scores_df'].copy()
    sample_group_scores_df['sample_idx'] = sample_group_scores_df['sample_idx'].astype(int)
    sample_group_scores_df['group_id'] = sample_group_scores_df['group_id'].astype(int)

    global_group_metrics_df = compute_global_group_risk(sample_group_scores_df, args)
    metadata_view = _build_tail_metadata_view(tgm_metadata_df)

    merged_df = sample_group_scores_df.merge(
        global_group_metrics_df,
        on=['group_id'],
        how='left',
        validate='many_to_one',
    )
    merged_df = merged_df.merge(
        metadata_view,
        on=['sample_idx', 'group_id', 'img_path'],
        how='left',
        validate='one_to_one',
    )
    if merged_df['tail_candidate'].isna().any():
        missing_count = int(merged_df['tail_candidate'].isna().sum())
        raise ValueError('missing tgm metadata for {} scored samples'.format(missing_count))

    merged_df['tail_candidate'] = merged_df['tail_candidate'].astype(int)
    merged_df['pred_class_size'] = merged_df['pred_class_size'].astype(float)
    merged_df['tail_score'] = merged_df['tail_score'].astype(float)
    merged_df['local_risk'] = (
        merged_df['q_g'].fillna(0.0).astype(float) * merged_df['p_high'].fillna(0.0).astype(float)
    )
    merged_df['global_group_risk'] = merged_df['global_group_risk'].fillna(0.0).astype(float)
    merged_df['final_gbps_risk'] = np.maximum(
        merged_df['local_risk'].to_numpy(dtype=float),
        merged_df['global_group_risk'].to_numpy(dtype=float),
    )

    summary = dict(gbps_result['summary'])
    summary.update({
        'tgm_local_risk_mean': float(merged_df['local_risk'].mean()) if len(merged_df) > 0 else 0.0,
        'tgm_global_group_risk_mean': float(merged_df['global_group_risk'].mean()) if len(merged_df) > 0 else 0.0,
        'tgm_final_risk_mean': float(merged_df['final_gbps_risk'].mean()) if len(merged_df) > 0 else 0.0,
        'tgm_num_tail_candidates': int(merged_df['tail_candidate'].sum()),
        'tgm_num_global_active_groups': int(global_group_metrics_df['global_is_active'].astype(bool).sum()) if len(global_group_metrics_df) > 0 else 0,
    })

    return {
        'gbps_U': gbps_result['gbps_U'],
        'gbps_SE': gbps_result['gbps_SE'],
        'group_metrics_df': gbps_result['group_metrics_df'],
        'sample_group_scores_df': sample_group_scores_df,
        'bootstrap_df': gbps_result['bootstrap_df'],
        'global_noise_evidence': gbps_result['global_noise_evidence'],
        'global_group_metrics_df': global_group_metrics_df,
        'sample_tgm_scores_df': merged_df,
        'summary': summary,
    }


def _iter_number_from_name(name: str):
    match = re.match(r'^iter_(\d+)$', str(name))
    if match is None:
        return None
    return int(match.group(1))


def aggregate_tgm_risk_history(diag_save_dir: str,
                                     selected_iter: Optional[int],
                                     lookback_checks: int = 3) -> pd.DataFrame:
    if selected_iter is None:
        return pd.DataFrame(columns=[
            'sample_idx', 'aggregated_local_risk', 'aggregated_global_group_risk',
            'aggregated_final_gbps_risk', 'num_risk_observations', 'risk_source_iters'
        ])

    iter_numbers = []
    for name in os.listdir(diag_save_dir):
        iter_number = _iter_number_from_name(name)
        if iter_number is None or iter_number > int(selected_iter):
            continue
        score_path = os.path.join(diag_save_dir, name, 'sample_tgm_scores.csv')
        if os.path.isfile(score_path):
            iter_numbers.append(iter_number)
    iter_numbers = sorted(iter_numbers)
    if len(iter_numbers) == 0:
        return pd.DataFrame(columns=[
            'sample_idx', 'aggregated_local_risk', 'aggregated_global_group_risk',
            'aggregated_final_gbps_risk', 'num_risk_observations', 'risk_source_iters'
        ])

    lookback_checks = max(1, int(lookback_checks))
    selected_iters = iter_numbers[-lookback_checks:]
    frames = []
    for iter_number in selected_iters:
        path = os.path.join(diag_save_dir, 'iter_{:05d}'.format(iter_number), 'sample_tgm_scores.csv')
        frame = pd.read_csv(path)
        frame['risk_iter'] = int(iter_number)
        frames.append(frame[['sample_idx', 'local_risk', 'global_group_risk', 'final_gbps_risk', 'risk_iter']].copy())
    history_df = pd.concat(frames, ignore_index=True)
    history_df['sample_idx'] = history_df['sample_idx'].astype(int)

    aggregated_df = history_df.groupby('sample_idx', sort=True).agg(
        aggregated_local_risk=('local_risk', 'median'),
        aggregated_global_group_risk=('global_group_risk', 'median'),
        aggregated_final_gbps_risk=('final_gbps_risk', 'median'),
        num_risk_observations=('risk_iter', 'count'),
    ).reset_index()
    iter_map = history_df.groupby('sample_idx')['risk_iter'].apply(lambda values: ','.join(str(int(value)) for value in sorted(values.unique()))).to_dict()
    aggregated_df['risk_source_iters'] = aggregated_df['sample_idx'].map(iter_map)
    return aggregated_df


def _build_baseline_remove_map(baseline_plan: Dict):
    if len(baseline_plan['pruned_samples']) == 0:
        return set()
    return set(baseline_plan['pruned_samples']['sample_idx'].astype(int).tolist())


def _build_retained_index_map(kept_samples: pd.DataFrame):
    retained_index_map = {}
    for class_id, class_df in kept_samples.groupby('class_id', sort=True):
        retained_index_map[int(class_id)] = [int(base_idx) for base_idx in class_df['base_idx'].tolist()]
    return retained_index_map


def build_tgm_delete_plan(scored_df: pd.DataFrame,
                                group_assignments_df: pd.DataFrame,
                                tgm_metadata_df: pd.DataFrame,
                                aggregated_risk_df: Optional[pd.DataFrame],
                                args) -> Dict:
    baseline_plan = build_gbps_hard_delete_plan(
        scored_df.copy(),
        group_assignments_df.copy(),
        prune_ratio=float(getattr(args, 'gbps_prune_ratio', 0.1)),
        min_keep_per_group=int(getattr(args, 'gbps_min_keep_per_group', 8)),
    )
    baseline_removed = _build_baseline_remove_map(baseline_plan)

    metadata_view = _build_tail_metadata_view(tgm_metadata_df)
    merged_df = baseline_plan['merged_scores_df'].copy()
    merged_df['sample_idx'] = merged_df['sample_idx'].astype(int)
    merged_df['group_id'] = merged_df['group_id'].astype(int)
    merged_df = merged_df.merge(
        metadata_view[['sample_idx', 'tail_candidate', 'pred_class_size', 'tail_score']],
        on='sample_idx',
        how='left',
        validate='one_to_one',
    )
    if aggregated_risk_df is None or len(aggregated_risk_df) == 0:
        aggregated_risk_df = merged_df[['sample_idx']].copy()
        aggregated_risk_df['aggregated_local_risk'] = 0.0
        aggregated_risk_df['aggregated_global_group_risk'] = 0.0
        aggregated_risk_df['aggregated_final_gbps_risk'] = 0.0
        aggregated_risk_df['num_risk_observations'] = 0
        aggregated_risk_df['risk_source_iters'] = ''
    else:
        aggregated_risk_df = aggregated_risk_df.copy()
        aggregated_risk_df['sample_idx'] = aggregated_risk_df['sample_idx'].astype(int)

    merged_df = merged_df.merge(
        aggregated_risk_df,
        on='sample_idx',
        how='left',
        validate='one_to_one',
    )
    merged_df['tail_candidate'] = merged_df['tail_candidate'].fillna(0).astype(int)
    merged_df['pred_class_size'] = merged_df['pred_class_size'].fillna(0.0).astype(float)
    merged_df['tail_score'] = merged_df['tail_score'].fillna(0.0).astype(float)
    merged_df['aggregated_local_risk'] = merged_df['aggregated_local_risk'].fillna(0.0).astype(float)
    merged_df['aggregated_global_group_risk'] = merged_df['aggregated_global_group_risk'].fillna(0.0).astype(float)
    merged_df['aggregated_final_gbps_risk'] = merged_df['aggregated_final_gbps_risk'].fillna(0.0).astype(float)
    merged_df['num_risk_observations'] = merged_df['num_risk_observations'].fillna(0).astype(int)
    merged_df['risk_source_iters'] = merged_df['risk_source_iters'].fillna('').astype(str)

    merged_df['baseline_remove'] = merged_df['sample_idx'].isin(baseline_removed)
    risk_thr = float(getattr(args, 'tgm_tail_risk_remove_thr', 0.5))
    merged_df['tgm_remove'] = merged_df['baseline_remove']
    tail_mask = merged_df['tail_candidate'] == 1
    merged_df.loc[tail_mask, 'tgm_remove'] = (
        merged_df.loc[tail_mask, 'baseline_remove']
        & (merged_df.loc[tail_mask, 'aggregated_final_gbps_risk'] >= risk_thr)
    )

    pruned_samples = merged_df.loc[merged_df['tgm_remove']].copy().reset_index(drop=True)
    kept_samples = merged_df.loc[~merged_df['tgm_remove']].copy().reset_index(drop=True)
    retained_index_map = _build_retained_index_map(kept_samples)

    tail_support_mask = (
        (merged_df['tail_candidate'] == 1)
        & (~merged_df['tgm_remove'])
        & (merged_df['aggregated_final_gbps_risk'] < risk_thr)
    )
    tail_suspicious_mask = (
        (merged_df['tail_candidate'] == 1)
        & (~tail_support_mask)
    )
    tail_support_samples = merged_df.loc[tail_support_mask].copy().reset_index(drop=True)
    tail_suspicious_samples = merged_df.loc[tail_suspicious_mask].copy().reset_index(drop=True)

    summary = {
        'mode': 'remove',
        'num_samples_before_prune': int(len(merged_df)),
        'num_pruned': int(len(pruned_samples)),
        'num_retained': int(len(kept_samples)),
        'prune_ratio': float(getattr(args, 'gbps_prune_ratio', 0.1)),
        'min_keep_per_group': int(getattr(args, 'gbps_min_keep_per_group', 8)),
        'num_groups': int(merged_df['group_id'].nunique()),
        'tgm_tail_risk_remove_thr': risk_thr,
        'num_baseline_removed': int(merged_df['baseline_remove'].sum()),
        'num_tail_candidates': int((merged_df['tail_candidate'] == 1).sum()),
        'num_tail_removed': int(((merged_df['tail_candidate'] == 1) & (merged_df['tgm_remove'])).sum()),
        'num_tail_retained': int(((merged_df['tail_candidate'] == 1) & (~merged_df['tgm_remove'])).sum()),
        'num_tail_support': int(len(tail_support_samples)),
        'num_tail_suspicious': int(len(tail_suspicious_samples)),
    }

    return {
        'mode': 'remove',
        'baseline_plan': baseline_plan,
        'pruned_samples': pruned_samples,
        'kept_samples': kept_samples,
        'retained_index_map': retained_index_map,
        'tail_support_samples': tail_support_samples,
        'tail_suspicious_samples': tail_suspicious_samples,
        'tgm_risk_table': merged_df,
        'summary': summary,
    }
