import os
import random
import re
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from gbps_runner_utils import build_gbps_hard_delete_plan
from safe_gbps import compute_gbps_u_with_bootstrap


def _iter_number_from_name(name: str):
    match = re.match(r'^iter_(\d+)$', str(name))
    if match is None:
        return None
    return int(match.group(1))


def _build_retained_index_map(samples_df: pd.DataFrame):
    retained_index_map = {}
    if len(samples_df) == 0:
        return retained_index_map
    for class_id, class_df in samples_df.groupby('class_id', sort=True):
        retained_index_map[int(class_id)] = [int(value) for value in class_df['base_idx'].astype(int).tolist()]
    return retained_index_map


def _metadata_view(metadata_df: pd.DataFrame):
    columns = [
        'sample_idx', 'sample_key', 'img_path', 'base_idx', 'tail_candidate', 'pred_class_size',
        'tail_score', 'adaptive_angle', 'group_id', 'group_size',
    ]
    optional_columns = ['class_id', 'class_name', 'is_contaminated']
    for column in optional_columns:
        if column in metadata_df.columns:
            columns.append(column)
    view = metadata_df[columns].copy()
    view['sample_idx'] = view['sample_idx'].astype(int)
    view['tail_candidate'] = view['tail_candidate'].astype(int)
    view['group_id'] = view['group_id'].astype(int)
    return view


def enrich_scores_with_tailguard_b_metadata(scored_df: pd.DataFrame, metadata_df: pd.DataFrame):
    scored = scored_df.copy()
    scored['sample_idx'] = scored['sample_idx'].astype(int)
    metadata_view = _metadata_view(metadata_df)
    drop_columns = [column for column in metadata_view.columns if column in scored.columns and column != 'sample_idx']
    scored = scored.drop(columns=drop_columns, errors='ignore')
    scored = scored.merge(metadata_view, on='sample_idx', how='left', validate='one_to_one')
    if scored['tail_candidate'].isna().any():
        missing_count = int(scored['tail_candidate'].isna().sum())
        raise ValueError('missing TailGuard-B metadata for {} scored samples'.format(missing_count))
    scored['tail_candidate'] = scored['tail_candidate'].astype(int)
    scored['group_id'] = scored['group_id'].astype(int)
    return scored


def run_tailguard_b_gbps_iteration(scored_df: pd.DataFrame,
                                   head_group_assignments_df: pd.DataFrame,
                                   metadata_df: pd.DataFrame,
                                   args):
    train_scores_df = enrich_scores_with_tailguard_b_metadata(scored_df, metadata_df)
    h_scores_df = train_scores_df.loc[train_scores_df['tail_candidate'] == 0].copy().reset_index(drop=True)
    if len(h_scores_df) == 0:
        raise ValueError('TailGuard-B GBPS requires non-empty H scores')

    gbps_result = compute_gbps_u_with_bootstrap(
        h_scores_df.copy(),
        head_group_assignments_df.copy(),
        B=int(getattr(args, 'gbps_bootstrap_B', 20)),
        args=args,
    )
    summary = dict(gbps_result['summary'])
    summary.update({
        'tailguard_b_h_only': True,
        'num_scored_total': int(len(train_scores_df)),
        'num_scored_h': int(len(h_scores_df)),
        'num_scored_t_saved_only': int((train_scores_df['tail_candidate'] == 1).sum()),
    })
    return {
        'gbps_U': gbps_result['gbps_U'],
        'gbps_SE': gbps_result['gbps_SE'],
        'global_noise_evidence': gbps_result['global_noise_evidence'],
        'train_scores_df': train_scores_df,
        'h_scores_df': h_scores_df,
        'h_group_metrics_df': gbps_result['group_metrics_df'],
        'h_sample_group_scores_df': gbps_result['sample_group_scores_df'],
        'bootstrap_df': gbps_result['bootstrap_df'],
        'summary': summary,
    }


def save_tailguard_b_selected_checkpoint(path: str,
                                         model,
                                         optimizer,
                                         scheduler,
                                         iteration: int,
                                         gbps_U: float,
                                         gbps_SE: float,
                                         args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'iteration': int(iteration),
        'gbps_U': float(gbps_U),
        'gbps_SE': float(gbps_SE),
        'rng_state': {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy': np.random.get_state(),
            'python': random.getstate(),
        },
        'args': vars(args).copy(),
    }
    torch.save(payload, path)
    return path


def load_tailguard_b_selected_checkpoint(path: str, model, optimizer=None, scheduler=None, map_location=None, restore_rng: bool = True):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and checkpoint.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if restore_rng and checkpoint.get('rng_state') is not None:
        rng_state = checkpoint['rng_state']
        if rng_state.get('torch') is not None:
            torch.set_rng_state(rng_state['torch'])
        if torch.cuda.is_available() and rng_state.get('cuda') is not None:
            torch.cuda.set_rng_state_all(rng_state['cuda'])
        if rng_state.get('numpy') is not None:
            np.random.set_state(rng_state['numpy'])
        if rng_state.get('python') is not None:
            random.setstate(rng_state['python'])
    return checkpoint


def build_tailguard_b_head_delete_plan(selected_scored_df: pd.DataFrame,
                                       head_group_assignments_df: pd.DataFrame,
                                       metadata_df: pd.DataFrame,
                                       args):
    selected_scores = enrich_scores_with_tailguard_b_metadata(selected_scored_df, metadata_df)
    h_scores_df = selected_scores.loc[selected_scores['tail_candidate'] == 0].copy().reset_index(drop=True)
    baseline_plan = build_gbps_hard_delete_plan(
        h_scores_df,
        head_group_assignments_df.copy(),
        prune_ratio=float(getattr(args, 'gbps_prune_ratio', 0.1)),
        min_keep_per_group=int(getattr(args, 'gbps_min_keep_per_group', 8)),
    )
    h_clean_df = baseline_plan['kept_samples'].copy().reset_index(drop=True)
    h_removed_df = baseline_plan['pruned_samples'].copy().reset_index(drop=True)
    summary = dict(baseline_plan['summary'])
    summary.update({
        'tailguard_b_h_only_removal': True,
        'num_h_clean': int(len(h_clean_df)),
        'num_h_removed': int(len(h_removed_df)),
        'num_t_untouched_by_h_removal': int((selected_scores['tail_candidate'] == 1).sum()),
    })
    return {
        'mode': 'h_only_remove',
        'selected_scores_df': selected_scores,
        'h_scores_df': h_scores_df,
        'h_clean_samples': h_clean_df,
        'h_removed_samples': h_removed_df,
        'retained_index_map_h_only': _build_retained_index_map(h_clean_df),
        'baseline_plan': baseline_plan,
        'summary': summary,
    }


def _list_window_iters(gbps_dir: str, selected_iter: int, stable_window: int):
    iter_numbers = []
    for name in os.listdir(gbps_dir):
        iter_number = _iter_number_from_name(name)
        if iter_number is not None and iter_number <= int(selected_iter):
            if os.path.isfile(os.path.join(gbps_dir, name, 'train_scores.csv')):
                iter_numbers.append(iter_number)
    iter_numbers = sorted(iter_numbers)
    return iter_numbers[-max(1, int(stable_window)):]


def _normal_pdf(x, mu, sigma):
    sigma = max(float(sigma), 1e-6)
    return np.exp(-0.5 * ((float(x) - float(mu)) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def _posterior_high_from_metrics(image_score: float, metrics_row: pd.Series):
    required = ['score_median', 'score_mad', 'mu_low', 'mu_high', 'sigma_low', 'sigma_high', 'pi_low', 'pi_high']
    if any(pd.isna(metrics_row.get(column)) for column in required):
        return None
    z_score = (float(image_score) - float(metrics_row['score_median'])) / max(float(metrics_row['score_mad']), 1e-6)
    low = float(metrics_row['pi_low']) * _normal_pdf(z_score, metrics_row['mu_low'], metrics_row['sigma_low'])
    high = float(metrics_row['pi_high']) * _normal_pdf(z_score, metrics_row['mu_high'], metrics_row['sigma_high'])
    denom = low + high
    if denom <= 0:
        return None
    return float(high / denom)


def classify_tail_attached_stable_risk(tail_attached_df: pd.DataFrame,
                                       gbps_dir: str,
                                       selected_iter: Optional[int],
                                       args):
    attached_df = tail_attached_df.copy().reset_index(drop=True)
    stable_columns = {
        'stable_p_high_median': 0.0,
        'stable_active_ratio': 0.0,
        'stable_valid_observations': 0,
        'stable_risk_source_iters': '',
        'stable_selected_group_active': False,
        'stable_head_noise': False,
    }
    if len(attached_df) == 0:
        for column, value in stable_columns.items():
            attached_df[column] = value
        return attached_df.copy(), attached_df.copy(), attached_df.copy()
    if selected_iter is None:
        for column, value in stable_columns.items():
            attached_df[column] = value
        return attached_df.copy(), attached_df.iloc[0:0].copy(), attached_df.copy()

    window_iters = _list_window_iters(gbps_dir, int(selected_iter), int(getattr(args, 'tgb_stable_window', 3)))
    p_high_thr = float(getattr(args, 'tgb_t_attached_noise_p_high_thr', 0.5))
    min_valid = int(getattr(args, 'tgb_stable_min_observations', 1))
    rows = []

    for _, row in attached_df.iterrows():
        sample_idx = int(row['sample_idx'])
        best_group_id = int(row['best_group_id'])
        p_high_values = []
        active_values = []
        source_iters = []
        selected_active = False
        for iter_number in window_iters:
            iter_dir = os.path.join(gbps_dir, 'iter_{:05d}'.format(int(iter_number)))
            score_path = os.path.join(iter_dir, 'train_scores.csv')
            metrics_path = os.path.join(iter_dir, 'h_group_metrics.csv')
            if not os.path.isfile(score_path) or not os.path.isfile(metrics_path):
                continue
            train_scores_df = pd.read_csv(score_path)
            group_metrics_df = pd.read_csv(metrics_path)
            score_rows = train_scores_df.loc[train_scores_df['sample_idx'].astype(int) == sample_idx]
            metric_rows = group_metrics_df.loc[group_metrics_df['group_id'].astype(int) == best_group_id]
            if len(score_rows) == 0 or len(metric_rows) == 0:
                continue
            metric_row = metric_rows.iloc[0]
            p_high = _posterior_high_from_metrics(float(score_rows.iloc[0]['image_score']), metric_row)
            if p_high is None:
                continue
            is_active = bool(metric_row.get('is_active_group', False))
            if int(iter_number) == int(selected_iter):
                selected_active = is_active
            p_high_values.append(float(p_high))
            active_values.append(1.0 if is_active else 0.0)
            source_iters.append(int(iter_number))

        valid_observations = len(p_high_values)
        active_ratio = float(np.mean(active_values)) if len(active_values) > 0 else 0.0
        p_high_median = float(np.median(p_high_values)) if len(p_high_values) > 0 else 0.0
        is_noise = bool(
            valid_observations >= min_valid
            and selected_active
            and active_ratio > 0.5
            and p_high_median > p_high_thr
        )
        out_row = row.to_dict()
        out_row.update({
            'stable_p_high_median': p_high_median,
            'stable_active_ratio': active_ratio,
            'stable_valid_observations': int(valid_observations),
            'stable_risk_source_iters': ','.join(str(value) for value in source_iters),
            'stable_selected_group_active': bool(selected_active),
            'stable_head_noise': bool(is_noise),
        })
        rows.append(out_row)

    risk_df = pd.DataFrame(rows)
    if len(risk_df) == 0:
        for column, value in stable_columns.items():
            attached_df[column] = value
        return attached_df.copy(), attached_df.iloc[0:0].copy(), attached_df.copy()
    noise_df = risk_df.loc[risk_df['stable_head_noise']].copy().reset_index(drop=True)
    normal_df = risk_df.loc[~risk_df['stable_head_noise']].copy().reset_index(drop=True)
    return normal_df, noise_df, risk_df


def build_tailguard_b_stage2_plan(h_clean_df: pd.DataFrame,
                                  h_removed_df: pd.DataFrame,
                                  tail_open_df: pd.DataFrame,
                                  tail_head_normal_df: pd.DataFrame,
                                  tail_head_noise_df: pd.DataFrame):
    retained_frames = [frame for frame in [h_clean_df, tail_head_normal_df, tail_open_df] if frame is not None and len(frame) > 0]
    removed_frames = [frame for frame in [h_removed_df, tail_head_noise_df] if frame is not None and len(frame) > 0]
    retained_df = pd.concat(retained_frames, ignore_index=True, sort=False) if len(retained_frames) > 0 else pd.DataFrame()
    removed_df = pd.concat(removed_frames, ignore_index=True, sort=False) if len(removed_frames) > 0 else pd.DataFrame()
    if len(retained_df) > 0:
        retained_df = retained_df.drop_duplicates(subset=['sample_idx']).reset_index(drop=True)
    if len(removed_df) > 0:
        removed_df = removed_df.drop_duplicates(subset=['sample_idx']).reset_index(drop=True)
    summary = {
        'num_stage2_retained': int(len(retained_df)),
        'num_stage2_removed': int(len(removed_df)),
        'num_h_clean': int(len(h_clean_df)),
        'num_h_removed': int(len(h_removed_df)),
        'num_tail_open': int(len(tail_open_df)),
        'num_tail_head_normal': int(len(tail_head_normal_df)),
        'num_tail_head_noise': int(len(tail_head_noise_df)),
    }
    return {
        'retained_samples': retained_df,
        'removed_samples': removed_df,
        'retained_index_map': _build_retained_index_map(retained_df),
        'summary': summary,
    }
