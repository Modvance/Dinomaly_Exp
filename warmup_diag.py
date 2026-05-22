import json
import math
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.mixture import GaussianMixture

from utils import compute_image_level_scores, get_gaussian_kernel, infer_anomaly_map_batch


def parse_warmup_milestones(milestones):
    values = []
    for part in milestones.split(','):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return sorted(set(values))


def _safe_float(value):
    if value is None:
        return None
    if isinstance(value, (np.floating, float)) and np.isnan(value):
        return None
    return float(value)


def _safe_mean(series):
    if len(series) == 0:
        return float('nan')
    return float(series.mean())


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


def _append_manifest_candidate(candidates, seen, candidate):
    candidate = Path(candidate)
    candidate_str = str(candidate)
    if candidate_str not in seen:
        seen.add(candidate_str)
        candidates.append(candidate)


def _extract_scheme_and_seed(name):
    direct_match = re.match(r'^(step_k\d+|pareto)[-_](seed\d+)$', name)
    if direct_match is not None:
        return direct_match.group(1), direct_match.group(2)

    visa_match = re.match(r'^visa-(step_k\d+|pareto)-(seed\d+)$', name)
    if visa_match is not None:
        return visa_match.group(1), visa_match.group(2)

    return None, None


def _resolve_manifest_candidates(data_path, repo_root):
    data_path = os.path.abspath(data_path)
    data_parts = Path(data_path).parts
    candidates = []
    seen = set()
    manifest_roots = [
        Path(repo_root) / 'manifest' / 'mvtecad-nlt',
        Path(repo_root) / 'manifest' / 'visa-nlt',
    ]

    for index, part in enumerate(data_parts):
        scheme, seed = _extract_scheme_and_seed(part)
        if scheme is not None and seed is not None:
            for manifest_root in manifest_roots:
                _append_manifest_candidate(candidates, seen, manifest_root / scheme / seed / 'inject_defects.txt')

        if part.startswith('step_k') or part == 'pareto':
            if index + 1 < len(data_parts) and data_parts[index + 1].startswith('seed'):
                for manifest_root in manifest_roots:
                    _append_manifest_candidate(candidates, seen, manifest_root / part / data_parts[index + 1] / 'inject_defects.txt')

    data_name = os.path.basename(os.path.normpath(data_path))
    scheme, seed = _extract_scheme_and_seed(data_name)
    if scheme is not None and seed is not None:
        for manifest_root in manifest_roots:
            _append_manifest_candidate(candidates, seen, manifest_root / scheme / seed / 'inject_defects.txt')

    if os.path.isfile(data_path):
        _append_manifest_candidate(candidates, seen, data_path)

    return candidates


def load_injected_manifest(data_path, repo_root, manifest_path=None):
    candidates = []
    if manifest_path is not None:
        candidates.append(Path(manifest_path))
    candidates.extend(_resolve_manifest_candidates(data_path, repo_root))

    for candidate in candidates:
        if candidate.is_file():
            with open(candidate) as file:
                paths = {line.strip().replace('\\', '/') for line in file if line.strip()}
            return paths, str(candidate)
    return None, None


def score_trainset(model, loader, device, max_ratio=0.01, resize_mask=256):
    rows = []
    was_training = model.training
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    model.eval()
    with torch.no_grad():
        for images, _, meta in loader:
            images = images.to(device)
            anomaly_map, _ = infer_anomaly_map_batch(model, images, gaussian_kernel, resize_mask=resize_mask)
            image_scores = compute_image_level_scores(anomaly_map, max_ratio=max_ratio).detach().cpu().numpy()

            batch_size = len(image_scores)
            for index in range(batch_size):
                row = {}
                for key, value in meta.items():
                    item = value[index]
                    if torch.is_tensor(item):
                        item = item.item()
                    elif isinstance(item, np.generic):
                        item = item.item()
                    row[key] = item
                row['image_score'] = float(image_scores[index])
                rows.append(row)

    if was_training:
        model.train()

    return pd.DataFrame(rows)


def compute_warmup_diagnostics(df):
    summary = {
        'num_samples': int(len(df)),
    }

    has_contamination = 'is_contaminated' in df.columns and df['is_contaminated'].isin([0, 1]).any()
    clean_mask = df['is_contaminated'] == 0 if has_contamination else None
    noisy_mask = df['is_contaminated'] == 1 if has_contamination else None

    summary['num_clean'] = int(clean_mask.sum()) if has_contamination else None
    summary['num_noisy'] = int(noisy_mask.sum()) if has_contamination else None

    class_metrics = {}
    macro_clean_means = []
    macro_noisy_means = []
    macro_score_gaps = []
    macro_aurocs = []
    macro_aps = []

    if 'class_name' in df.columns:
        for class_name, class_df in df.groupby('class_name'):
            clean_mean = _safe_mean(class_df.loc[class_df['is_contaminated'] == 0, 'image_score']) if has_contamination else float('nan')
            noisy_mean = _safe_mean(class_df.loc[class_df['is_contaminated'] == 1, 'image_score']) if has_contamination else float('nan')
            score_gap = noisy_mean - clean_mean
            class_auroc = _safe_binary_metric(roc_auc_score, class_df['is_contaminated'], class_df['image_score']) if has_contamination else float('nan')
            class_ap = _safe_binary_metric(average_precision_score, class_df['is_contaminated'], class_df['image_score']) if has_contamination else float('nan')

            class_metrics[str(class_name)] = {
                'num_samples': int(len(class_df)),
                'clean_mean_score': _safe_float(clean_mean),
                'noisy_mean_score': _safe_float(noisy_mean),
                'score_gap': _safe_float(score_gap),
                'train_noise_auroc': _safe_float(class_auroc),
                'train_noise_ap': _safe_float(class_ap),
            }

            if not np.isnan(clean_mean):
                macro_clean_means.append(clean_mean)
            if not np.isnan(noisy_mean):
                macro_noisy_means.append(noisy_mean)
            if not np.isnan(score_gap):
                macro_score_gaps.append(score_gap)
            if not np.isnan(class_auroc):
                macro_aurocs.append(class_auroc)
            if not np.isnan(class_ap):
                macro_aps.append(class_ap)

    if has_contamination:
        summary.update({
            'clean_mean_score': _safe_float(_safe_mean(pd.Series(macro_clean_means))) if len(macro_clean_means) > 0 else None,
            'noisy_mean_score': _safe_float(_safe_mean(pd.Series(macro_noisy_means))) if len(macro_noisy_means) > 0 else None,
            'score_gap': _safe_float(_safe_mean(pd.Series(macro_score_gaps))) if len(macro_score_gaps) > 0 else None,
            'train_noise_auroc': _safe_float(_safe_mean(pd.Series(macro_aurocs))) if len(macro_aurocs) > 0 else None,
            'train_noise_ap': _safe_float(_safe_mean(pd.Series(macro_aps))) if len(macro_aps) > 0 else None,
            'metric_averaging': 'macro_over_classes',
        })
    else:
        summary.update({
            'clean_mean_score': None,
            'noisy_mean_score': None,
            'score_gap': None,
            'train_noise_auroc': None,
            'train_noise_ap': None,
            'metric_averaging': 'macro_over_classes',
        })

    summary['class_metrics'] = class_metrics

    return summary


def plot_global_hist(df, save_path):
    if 'is_contaminated' not in df.columns or not df['is_contaminated'].isin([0, 1]).any():
        return False

    plt.figure(figsize=(8, 5))
    clean_scores = df.loc[df['is_contaminated'] == 0, 'image_score']
    noisy_scores = df.loc[df['is_contaminated'] == 1, 'image_score']
    if len(clean_scores) > 0:
        plt.hist(clean_scores, bins=30, alpha=0.6, label='clean')
    if len(noisy_scores) > 0:
        plt.hist(noisy_scores, bins=30, alpha=0.6, label='noisy')
    plt.xlabel('image_score')
    plt.ylabel('count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return True


def plot_class_gap(df, save_path):
    if 'class_name' not in df.columns or 'is_contaminated' not in df.columns:
        return False

    records = []
    for class_name, class_df in df.groupby('class_name'):
        clean_scores = class_df.loc[class_df['is_contaminated'] == 0, 'image_score']
        noisy_scores = class_df.loc[class_df['is_contaminated'] == 1, 'image_score']
        if len(clean_scores) == 0 or len(noisy_scores) == 0:
            continue
        records.append((str(class_name), float(noisy_scores.mean() - clean_scores.mean())))

    if len(records) == 0:
        return False

    names, gaps = zip(*records)
    plt.figure(figsize=(10, 5))
    positions = np.arange(len(names))
    plt.bar(positions, gaps)
    plt.xticks(positions, names, rotation=45, ha='right')
    plt.ylabel('score_gap')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return True


def plot_class_hists(df, save_dir):
    if 'class_name' not in df.columns or 'is_contaminated' not in df.columns:
        return []

    saved_paths = []
    class_hist_dir = os.path.join(save_dir, 'class_hists')
    os.makedirs(class_hist_dir, exist_ok=True)

    for class_name, class_df in df.groupby('class_name'):
        if not class_df['is_contaminated'].isin([0, 1]).any():
            continue

        clean_scores = class_df.loc[class_df['is_contaminated'] == 0, 'image_score']
        noisy_scores = class_df.loc[class_df['is_contaminated'] == 1, 'image_score']
        if len(clean_scores) == 0 and len(noisy_scores) == 0:
            continue

        plt.figure(figsize=(8, 5))
        if len(clean_scores) > 0:
            plt.hist(clean_scores, bins=30, alpha=0.6, label='clean')
        if len(noisy_scores) > 0:
            plt.hist(noisy_scores, bins=30, alpha=0.6, label='noisy')
        plt.title(str(class_name))
        plt.xlabel('image_score')
        plt.ylabel('count')
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(class_hist_dir, '{}_hist_global.png'.format(str(class_name)))
        plt.savefig(save_path)
        plt.close()
        saved_paths.append(save_path)

    return saved_paths


def _make_json_safe(value):
    if isinstance(value, dict):
        return {key: _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _require_identity_columns(df):
    required_columns = ['class_id', 'image_score', 'base_idx']
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError('missing required columns: {}'.format(', '.join(missing_columns)))


def _build_kept_df(df, removed_df):
    if len(removed_df) == 0:
        return df.copy()
    removed_keys = set(zip(removed_df['class_id'].tolist(), removed_df['base_idx'].tolist()))
    keep_mask = [
        (class_id, base_idx) not in removed_keys
        for class_id, base_idx in zip(df['class_id'].tolist(), df['base_idx'].tolist())
    ]
    return df.loc[keep_mask].copy().reset_index(drop=True)


def _build_weight_map(df, weight_col='sample_weight'):
    sample_weight_map = {}
    for _, row in df.iterrows():
        sample_weight_map[(int(row['class_id']), int(row['base_idx']))] = float(row[weight_col])
    return sample_weight_map


def _compute_group_weights(class_df, beta, min_weight, eps):
    class_df = class_df.copy()
    median = float(class_df['image_score'].median())
    mad = float(np.median(np.abs(class_df['image_score'].to_numpy() - median))) + float(eps)
    z_score = (class_df['image_score'] - median) / mad
    raw_weight = 1.0 / (1.0 + np.exp(float(beta) * z_score.to_numpy()))
    sample_weight = np.clip(raw_weight, float(min_weight), 1.0)
    class_df['score_median'] = median
    class_df['score_mad'] = mad
    class_df['z_score'] = z_score.astype(float)
    class_df['sample_weight'] = sample_weight.astype(float)
    return class_df


def _build_identity_keys(df):
    _require_identity_columns(df)
    return [
        (int(class_id), int(base_idx))
        for class_id, base_idx in zip(df['class_id'].tolist(), df['base_idx'].tolist())
    ]


def _compute_jaccard(keys_a, keys_b):
    if keys_a is None or keys_b is None:
        return None
    union = keys_a | keys_b
    if len(union) == 0:
        return 1.0
    return float(len(keys_a & keys_b) / len(union))


def _robust_normalize_scores(df, eps=1e-6):
    _require_identity_columns(df)
    df = df.copy()
    scores = df['image_score'].to_numpy(dtype=float)
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median))) + float(eps)
    z_score = (scores - median) / mad
    df['score_median'] = median
    df['score_mad'] = mad
    df['z_score'] = z_score.astype(float)
    return df, {'score_median': median, 'score_mad': mad}


def _fit_two_component_gmm(z_values, eps=1e-6, random_state=0):
    z_values = np.asarray(z_values, dtype=float).reshape(-1)
    if z_values.size == 0:
        raise ValueError('cannot fit GMM on empty scores')

    if z_values.size < 2 or np.allclose(z_values, z_values[0]):
        mean_value = float(np.mean(z_values))
        sigma_value = float(np.std(z_values) + eps)
        posterior_high = np.full(z_values.shape[0], 0.5, dtype=float)
        return {
            'mu_1': mean_value,
            'mu_2': mean_value,
            'sigma_1': sigma_value,
            'sigma_2': sigma_value,
            'weight_1': 0.5,
            'weight_2': 0.5,
            'D_t': 0.0,
            'posterior_high': posterior_high,
        }

    try:
        gmm = GaussianMixture(n_components=2, covariance_type='full', reg_covar=eps, random_state=random_state)
        gmm.fit(z_values.reshape(-1, 1))
        means = gmm.means_.reshape(-1)
        variances = gmm.covariances_.reshape(2, -1)[:, 0]
        sigmas = np.sqrt(np.maximum(variances, eps))
        weights = gmm.weights_.reshape(-1)
        order = np.argsort(means)
        low_idx, high_idx = int(order[0]), int(order[1])
        posterior_high = gmm.predict_proba(z_values.reshape(-1, 1))[:, high_idx]
        mu_1 = float(means[low_idx])
        mu_2 = float(means[high_idx])
        sigma_1 = float(sigmas[low_idx])
        sigma_2 = float(sigmas[high_idx])
        D_t = float((mu_2 - mu_1) / max(sigma_1 + sigma_2, eps))
        return {
            'mu_1': mu_1,
            'mu_2': mu_2,
            'sigma_1': sigma_1,
            'sigma_2': sigma_2,
            'weight_1': float(weights[low_idx]),
            'weight_2': float(weights[high_idx]),
            'D_t': D_t,
            'posterior_high': posterior_high.astype(float),
        }
    except Exception:
        mean_value = float(np.mean(z_values))
        sigma_value = float(np.std(z_values) + eps)
        posterior_high = np.full(z_values.shape[0], 0.5, dtype=float)
        return {
            'mu_1': mean_value,
            'mu_2': mean_value,
            'sigma_1': sigma_value,
            'sigma_2': sigma_value,
            'weight_1': 0.5,
            'weight_2': 0.5,
            'D_t': 0.0,
            'posterior_high': posterior_high,
        }


def annotate_scores_with_gmm(df, eps=1e-6, random_state=0):
    normalized_df, robust_stats = _robust_normalize_scores(df, eps=eps)
    gmm_stats = _fit_two_component_gmm(normalized_df['z_score'].to_numpy(dtype=float), eps=eps, random_state=random_state)
    annotated_df = normalized_df.copy()
    annotated_df['gmm_posterior_high'] = gmm_stats['posterior_high']
    annotated_df['gmm_posterior_low'] = 1.0 - annotated_df['gmm_posterior_high']
    stats = {
        'score_median': robust_stats['score_median'],
        'score_mad': robust_stats['score_mad'],
        'gmm_mu_1': gmm_stats['mu_1'],
        'gmm_mu_2': gmm_stats['mu_2'],
        'gmm_sigma_1': gmm_stats['sigma_1'],
        'gmm_sigma_2': gmm_stats['sigma_2'],
        'gmm_weight_1': gmm_stats['weight_1'],
        'gmm_weight_2': gmm_stats['weight_2'],
        'D_t': gmm_stats['D_t'],
    }
    return annotated_df, stats


def build_top_suspicious_samples(df, top_p):
    _require_identity_columns(df)
    if len(df) == 0:
        return df.copy(), set(), 0
    top_count = max(1, int(math.ceil(len(df) * float(top_p))))
    suspicious_df = df.sort_values('image_score', ascending=False).head(top_count).copy().reset_index(drop=True)
    suspicious_keys = set(_build_identity_keys(suspicious_df))
    return suspicious_df, suspicious_keys, top_count


def _evaluate_trigger_policy(trigger_mode, current_primary_value, best_primary_value, previous_best_primary_value,
                             no_improve_count, J_t, patience, plateau_ratio, jaccard_threshold,
                             force_trigger=False):
    patience_exhausted = int(no_improve_count) >= int(patience)
    plateau_ready = (
        previous_best_primary_value is not None
        and current_primary_value >= float(plateau_ratio) * float(best_primary_value)
        and patience_exhausted
        and J_t is not None
        and J_t >= float(jaccard_threshold)
    )
    peak_patience_ready = previous_best_primary_value is not None and patience_exhausted

    if trigger_mode == 'plateau':
        mode_triggered = plateau_ready
        trigger_reason = 'plateau' if plateau_ready else 'none'
        trigger_uses_aux_jaccard = True
        trigger_uses_plateau_ratio = True
    elif trigger_mode == 'peak_patience':
        mode_triggered = peak_patience_ready
        trigger_reason = 'peak_patience' if peak_patience_ready else 'none'
        trigger_uses_aux_jaccard = False
        trigger_uses_plateau_ratio = False
    else:
        raise ValueError(f'unsupported warm-up trigger mode: {trigger_mode}')

    if force_trigger:
        return {
            'triggered': True,
            'trigger_reason': 'forced',
            'force_triggered': True,
            'patience_exhausted': bool(patience_exhausted),
            'trigger_uses_aux_jaccard': trigger_uses_aux_jaccard,
            'trigger_uses_plateau_ratio': trigger_uses_plateau_ratio,
            'plateau_ready': bool(plateau_ready),
            'peak_patience_ready': bool(peak_patience_ready),
        }

    return {
        'triggered': bool(mode_triggered),
        'trigger_reason': trigger_reason,
        'force_triggered': False,
        'patience_exhausted': bool(patience_exhausted),
        'trigger_uses_aux_jaccard': trigger_uses_aux_jaccard,
        'trigger_uses_plateau_ratio': trigger_uses_plateau_ratio,
        'plateau_ready': bool(plateau_ready),
        'peak_patience_ready': bool(peak_patience_ready),
    }



def evaluate_warmup_trigger(scored_df, best_primary_value, best_primary_iter, no_improve_count,
                            last_suspicious_keys, current_iter, top_p, jaccard_threshold,
                            patience, plateau_ratio, primary_metric_name='D_t', trigger_mode='plateau', eps=1e-6,
                            random_state=0, force_trigger=False):
    annotated_df, gmm_stats = annotate_scores_with_gmm(scored_df, eps=eps, random_state=random_state)
    diagnosis_summary = compute_warmup_diagnostics(annotated_df)
    suspicious_df, suspicious_keys, top_count = build_top_suspicious_samples(annotated_df, top_p=top_p)
    J_t = _compute_jaccard(last_suspicious_keys, suspicious_keys)
    D_t = float(gmm_stats['D_t'])
    score_gap = diagnosis_summary.get('score_gap')

    if primary_metric_name == 'D_t':
        current_primary_value = D_t
    elif primary_metric_name == 'score_gap':
        current_primary_value = score_gap
        if current_primary_value is None:
            raise ValueError('warmup_primary_metric=score_gap requires contamination-aware score_gap metrics. Please provide a valid inject_defects manifest via --diag_manifest_path.')
        current_primary_value = float(current_primary_value)
    else:
        raise ValueError(f'unsupported warm-up primary metric: {primary_metric_name}')

    previous_best_primary_value = None if best_primary_value is None else float(best_primary_value)
    improved_primary_metric = previous_best_primary_value is None or current_primary_value > previous_best_primary_value
    if improved_primary_metric:
        best_primary_value = current_primary_value
        best_primary_iter = int(current_iter)
        no_improve_count = 0
    else:
        no_improve_count = int(no_improve_count) + 1

    trigger_eval = _evaluate_trigger_policy(
        trigger_mode=trigger_mode,
        current_primary_value=current_primary_value,
        best_primary_value=best_primary_value,
        previous_best_primary_value=previous_best_primary_value,
        no_improve_count=no_improve_count,
        J_t=J_t,
        patience=patience,
        plateau_ratio=plateau_ratio,
        jaccard_threshold=jaccard_threshold,
        force_trigger=force_trigger,
    )

    summary = {
        'stage': 'warmup',
        'iteration': int(current_iter),
        'top_p': float(top_p),
        'top_count': int(top_count),
        'D_t': D_t,
        'score_gap': _safe_float(score_gap),
        'J_t': _safe_float(J_t),
        'warmup_primary_metric': str(primary_metric_name),
        'warmup_trigger_mode': str(trigger_mode),
        'primary_metric_value': _safe_float(current_primary_value),
        'best_primary_value': _safe_float(best_primary_value),
        'best_primary_iter': None if best_primary_iter is None else int(best_primary_iter),
        'previous_best_primary_value': _safe_float(previous_best_primary_value),
        'no_improve_count': int(no_improve_count),
        'improved_primary_metric': bool(improved_primary_metric),
        'jaccard_threshold': float(jaccard_threshold),
        'patience': int(patience),
        'plateau_ratio': float(plateau_ratio),
        'triggered': bool(trigger_eval['triggered']),
        'trigger_reason': trigger_eval['trigger_reason'],
        'warmup_trigger_iter': int(current_iter),
        'force_triggered': bool(trigger_eval['force_triggered']),
        'patience_exhausted': bool(trigger_eval['patience_exhausted']),
        'trigger_uses_aux_jaccard': bool(trigger_eval['trigger_uses_aux_jaccard']),
        'trigger_uses_plateau_ratio': bool(trigger_eval['trigger_uses_plateau_ratio']),
        'plateau_ready': bool(trigger_eval['plateau_ready']),
        'peak_patience_ready': bool(trigger_eval['peak_patience_ready']),
        'best_D': _safe_float(best_primary_value) if primary_metric_name == 'D_t' else None,
        'previous_best_D': _safe_float(previous_best_primary_value) if primary_metric_name == 'D_t' else None,
        'improved_D': bool(improved_primary_metric) if primary_metric_name == 'D_t' else None,
    }
    summary.update(gmm_stats)

    return {
        'scored_df': annotated_df,
        'suspicious_df': suspicious_df,
        'suspicious_keys': suspicious_keys,
        'triggered': trigger_eval['triggered'],
        'trigger_reason': trigger_eval['trigger_reason'],
        'best_primary_value': best_primary_value,
        'best_primary_iter': best_primary_iter,
        'no_improve_count': no_improve_count,
        'current_primary_value': current_primary_value,
        'primary_metric_name': primary_metric_name,
        'summary': summary,
    }


def build_prune_plan(df, prune_ratio, min_keep_per_class):
    _require_identity_columns(df)

    pruned_frames = []
    retained_index_map = {}
    class_counts = {}

    grouped = df.groupby('class_id', sort=True)
    for class_id, class_df in grouped:
        class_df = class_df.sort_values('image_score', ascending=False).reset_index(drop=True)
        num_samples = int(len(class_df))
        requested_remove = int(math.floor(num_samples * float(prune_ratio)))
        max_removable = max(0, num_samples - int(min_keep_per_class))
        remove_count = min(requested_remove, max_removable)

        pruned_df = class_df.iloc[:remove_count].copy()
        retained_df = class_df.iloc[remove_count:].copy()

        retained_index_map[int(class_id)] = [int(base_idx) for base_idx in retained_df['base_idx'].tolist()]
        class_name = str(class_df['class_name'].iloc[0]) if 'class_name' in class_df.columns and num_samples > 0 else str(class_id)
        class_counts[str(class_name)] = {
            'class_id': int(class_id),
            'num_samples': num_samples,
            'removed': int(remove_count),
            'retained': int(len(retained_df)),
            'requested_remove': int(requested_remove),
        }
        if remove_count > 0:
            pruned_frames.append(pruned_df)

    pruned_samples = pd.concat(pruned_frames, ignore_index=True) if len(pruned_frames) > 0 else pd.DataFrame(columns=df.columns)
    kept_samples = _build_kept_df(df, pruned_samples)
    summary = {
        'mode': 'remove',
        'num_samples_before_prune': int(len(df)),
        'num_pruned': int(len(pruned_samples)),
        'num_retained': int(sum(len(indices) for indices in retained_index_map.values())),
        'prune_ratio': float(prune_ratio),
        'min_keep_per_class': int(min_keep_per_class),
        'class_counts': class_counts,
    }
    return {
        'mode': 'remove',
        'pruned_samples': pruned_samples,
        'kept_samples': kept_samples,
        'retained_index_map': retained_index_map,
        'sample_weight_map': {},
        'weighted_samples': pd.DataFrame(columns=df.columns),
        'summary': summary,
    }


def build_soft_plan(df, beta, min_weight, weight_scope='class', eps=1e-6):
    _require_identity_columns(df)

    if weight_scope == 'class':
        weighted_samples = pd.concat([
            _compute_group_weights(class_df, beta=beta, min_weight=min_weight, eps=eps)
            for _, class_df in df.groupby('class_id', sort=True)
        ], ignore_index=True)
    elif weight_scope == 'global':
        weighted_samples = _compute_group_weights(df, beta=beta, min_weight=min_weight, eps=eps)
    else:
        raise ValueError('unsupported weight_scope {}'.format(weight_scope))

    weighted_samples = weighted_samples.reset_index(drop=True)
    retained_index_map = {
        int(class_id): [int(base_idx) for base_idx in class_df['base_idx'].tolist()]
        for class_id, class_df in weighted_samples.groupby('class_id', sort=True)
    }
    summary = {
        'mode': 'soft',
        'weight_scope': weight_scope,
        'weight_beta': float(beta),
        'min_weight': float(min_weight),
        'num_samples_before_prune': int(len(df)),
        'num_pruned': 0,
        'num_retained': int(len(weighted_samples)),
        'weight_mean': _safe_float(weighted_samples['sample_weight'].mean()),
        'weight_std': _safe_float(weighted_samples['sample_weight'].std()),
        'weight_min': _safe_float(weighted_samples['sample_weight'].min()),
        'weight_max': _safe_float(weighted_samples['sample_weight'].max()),
    }
    return {
        'mode': 'soft',
        'pruned_samples': pd.DataFrame(columns=df.columns),
        'kept_samples': weighted_samples.copy(),
        'retained_index_map': retained_index_map,
        'sample_weight_map': _build_weight_map(weighted_samples),
        'weighted_samples': weighted_samples,
        'summary': summary,
    }


def build_hybrid_plan(df, prune_ratio, min_keep_per_class, beta, min_weight, weight_scope='class', eps=1e-6):
    prune_plan = build_prune_plan(df, prune_ratio=prune_ratio, min_keep_per_class=min_keep_per_class)
    kept_samples = prune_plan['kept_samples'].copy()
    soft_plan = build_soft_plan(kept_samples, beta=beta, min_weight=min_weight, weight_scope=weight_scope, eps=eps)
    summary = dict(soft_plan['summary'])
    summary.update({
        'mode': 'hybrid',
        'hybrid_prune_ratio': float(prune_ratio),
        'min_keep_per_class': int(min_keep_per_class),
        'num_samples_before_prune': int(len(df)),
        'num_pruned': int(len(prune_plan['pruned_samples'])),
        'num_retained': int(len(soft_plan['weighted_samples'])),
        'class_counts': prune_plan['summary']['class_counts'],
    })
    return {
        'mode': 'hybrid',
        'pruned_samples': prune_plan['pruned_samples'],
        'kept_samples': soft_plan['kept_samples'],
        'retained_index_map': prune_plan['retained_index_map'],
        'sample_weight_map': soft_plan['sample_weight_map'],
        'weighted_samples': soft_plan['weighted_samples'],
        'summary': summary,
    }


def build_warmup_postprocess_plan(df, mode, prune_ratio, min_keep_per_class, weight_beta, min_weight,
                                  weight_scope='class', eps=1e-6):
    if mode == 'none':
        return {
            'mode': 'none',
            'pruned_samples': pd.DataFrame(columns=df.columns),
            'kept_samples': df.copy(),
            'retained_index_map': {
                int(class_id): [int(base_idx) for base_idx in class_df['base_idx'].tolist()]
                for class_id, class_df in df.groupby('class_id', sort=True)
            },
            'sample_weight_map': {},
            'weighted_samples': pd.DataFrame(columns=df.columns),
            'summary': {
                'mode': 'none',
                'num_samples_before_prune': int(len(df)),
                'num_pruned': 0,
                'num_retained': int(len(df)),
            },
        }
    if mode == 'remove':
        return build_prune_plan(df, prune_ratio=prune_ratio, min_keep_per_class=min_keep_per_class)
    if mode == 'soft':
        return build_soft_plan(df, beta=weight_beta, min_weight=min_weight, weight_scope=weight_scope, eps=eps)
    if mode == 'hybrid':
        return build_hybrid_plan(
            df,
            prune_ratio=prune_ratio,
            min_keep_per_class=min_keep_per_class,
            beta=weight_beta,
            min_weight=min_weight,
            weight_scope=weight_scope,
            eps=eps,
        )
    raise ValueError('unsupported warmup postprocess mode {}'.format(mode))


def build_phase2_weight_update(scored_df, reliability_map, old_weight_map, min_weight, weight_beta,
                               weight_scope='class', reliability_ema=0.9, weight_ema=0.8,
                               clip_delta=0.1, top_p=0.1, eps=1e-6, random_state=0):
    annotated_df, gmm_stats = annotate_scores_with_gmm(scored_df, eps=eps, random_state=random_state)
    suspicious_df, suspicious_keys, top_count = build_top_suspicious_samples(annotated_df, top_p=top_p)
    soft_plan = build_soft_plan(annotated_df, beta=weight_beta, min_weight=min_weight, weight_scope=weight_scope, eps=eps)
    weighted_df = soft_plan['weighted_samples'].copy().reset_index(drop=True)

    new_reliability_map = {}
    new_sample_weight_map = {}
    reliability_values = []
    weight_values = []
    weight_deltas = []
    clipped_count = 0

    rows = []
    for _, row in weighted_df.iterrows():
        key = (int(row['class_id']), int(row['base_idx']))
        instant_reliability = float(np.clip(1.0 - row['gmm_posterior_high'], 0.0, 1.0))
        previous_reliability = reliability_map.get(key)
        if previous_reliability is None:
            reliability_value = instant_reliability
        else:
            reliability_value = float(reliability_ema) * float(previous_reliability) + (1.0 - float(reliability_ema)) * instant_reliability
        reliability_value = float(np.clip(reliability_value, 0.0, 1.0))

        score_weight = float(row['sample_weight'])
        instant_weight = float(min_weight) + (score_weight - float(min_weight)) * reliability_value
        previous_weight = old_weight_map.get(key)
        if previous_weight is None:
            blended_weight = instant_weight
            clipped_weight = instant_weight
            weight_delta = 0.0
        else:
            blended_weight = float(weight_ema) * float(previous_weight) + (1.0 - float(weight_ema)) * instant_weight
            if clip_delta is None:
                clipped_weight = blended_weight
            else:
                lower = float(previous_weight) - float(clip_delta)
                upper = float(previous_weight) + float(clip_delta)
                clipped_weight = float(np.clip(blended_weight, lower, upper))
                if abs(clipped_weight - blended_weight) > 1e-12:
                    clipped_count += 1
            weight_delta = abs(clipped_weight - float(previous_weight))

        final_weight = float(np.clip(clipped_weight, float(min_weight), 1.0))

        new_reliability_map[key] = reliability_value
        new_sample_weight_map[key] = final_weight
        reliability_values.append(reliability_value)
        weight_values.append(final_weight)
        weight_deltas.append(weight_delta)

        row_data = row.to_dict()
        row_data['instant_reliability'] = instant_reliability
        row_data['reliability'] = reliability_value
        row_data['previous_reliability'] = None if previous_reliability is None else float(previous_reliability)
        row_data['score_weight'] = score_weight
        row_data['instant_weight'] = instant_weight
        row_data['previous_weight'] = None if previous_weight is None else float(previous_weight)
        row_data['sample_weight'] = final_weight
        rows.append(row_data)

    final_weight_df = pd.DataFrame(rows)
    reliability_df = final_weight_df[[
        'class_id', 'class_name', 'base_idx', 'img_path', 'image_score', 'z_score', 'gmm_posterior_high',
        'instant_reliability', 'previous_reliability', 'reliability'
    ]].copy() if len(final_weight_df) > 0 else pd.DataFrame()

    summary = {
        'weight_scope': weight_scope,
        'weight_beta': float(weight_beta),
        'min_weight': float(min_weight),
        'top_p': float(top_p),
        'top_count': int(top_count),
        'num_samples': int(len(final_weight_df)),
        'reliability_mean': _safe_float(np.mean(reliability_values)) if len(reliability_values) > 0 else None,
        'reliability_std': _safe_float(np.std(reliability_values)) if len(reliability_values) > 0 else None,
        'reliability_min': _safe_float(np.min(reliability_values)) if len(reliability_values) > 0 else None,
        'reliability_max': _safe_float(np.max(reliability_values)) if len(reliability_values) > 0 else None,
        'weight_mean': _safe_float(np.mean(weight_values)) if len(weight_values) > 0 else None,
        'weight_std': _safe_float(np.std(weight_values)) if len(weight_values) > 0 else None,
        'weight_min': _safe_float(np.min(weight_values)) if len(weight_values) > 0 else None,
        'weight_max': _safe_float(np.max(weight_values)) if len(weight_values) > 0 else None,
        'weight_delta_mean_abs': _safe_float(np.mean(weight_deltas)) if len(weight_deltas) > 0 else None,
        'weight_delta_max_abs': _safe_float(np.max(weight_deltas)) if len(weight_deltas) > 0 else None,
        'clipped_sample_count': int(clipped_count),
    }
    summary.update(gmm_stats)

    return {
        'scored_df': annotated_df,
        'suspicious_df': suspicious_df,
        'suspicious_keys': suspicious_keys,
        'weight_df': final_weight_df,
        'reliability_df': reliability_df,
        'sample_weight_map': new_sample_weight_map,
        'reliability_map': new_reliability_map,
        'summary': summary,
    }


def save_prune_plan(iter_dir, prune_plan, current_iter):
    prune_summary = dict(prune_plan['summary'])
    prune_summary['iteration'] = int(current_iter)

    with open(os.path.join(iter_dir, 'prune_summary.json'), 'w') as file:
        json.dump(_make_json_safe(prune_summary), file, indent=2)

    if len(prune_plan['pruned_samples']) > 0:
        prune_plan['pruned_samples'].to_csv(os.path.join(iter_dir, 'removed_samples.csv'), index=False)
        prune_plan['pruned_samples'].to_csv(os.path.join(iter_dir, 'pruned_samples.csv'), index=False)
    if 'kept_samples' in prune_plan and len(prune_plan['kept_samples']) > 0:
        prune_plan['kept_samples'].to_csv(os.path.join(iter_dir, 'kept_samples.csv'), index=False)
    if 'weighted_samples' in prune_plan and len(prune_plan['weighted_samples']) > 0:
        prune_plan['weighted_samples'].to_csv(os.path.join(iter_dir, 'sample_weights.csv'), index=False)


def save_phase2_artifacts(iter_dir, extra_summary=None, suspicious_df=None, weight_df=None, reliability_df=None):
    summary_path = os.path.join(iter_dir, 'summary.json')
    summary = {}
    if os.path.isfile(summary_path):
        with open(summary_path) as file:
            summary = json.load(file)
    if extra_summary is not None:
        summary.update(_make_json_safe(extra_summary))
    with open(summary_path, 'w') as file:
        json.dump(_make_json_safe(summary), file, indent=2)

    if suspicious_df is not None and len(suspicious_df) > 0:
        suspicious_df.to_csv(os.path.join(iter_dir, 'suspicious_samples.csv'), index=False)
    if weight_df is not None and len(weight_df) > 0:
        weight_df.to_csv(os.path.join(iter_dir, 'sample_weights.csv'), index=False)
    if reliability_df is not None and len(reliability_df) > 0:
        reliability_df.to_csv(os.path.join(iter_dir, 'reliability_bank.csv'), index=False)


def load_saved_train_scores(save_dir, iteration):
    score_path = os.path.join(save_dir, 'iter_{:05d}'.format(int(iteration)), 'train_scores.csv')
    if not os.path.isfile(score_path):
        raise FileNotFoundError('saved train scores not found for iter {}: {}'.format(int(iteration), score_path))
    return pd.read_csv(score_path), score_path


def run_one_warmup_diagnosis(model, loader, device, save_dir, current_iter, print_fn=None, max_ratio=0.01,
                             resize_mask=256, manifest_path=None, save_scores=True):
    iter_dir = os.path.join(save_dir, 'iter_{:05d}'.format(current_iter))
    os.makedirs(iter_dir, exist_ok=True)

    df = score_trainset(model, loader, device, max_ratio=max_ratio, resize_mask=resize_mask)
    summary = compute_warmup_diagnostics(df)
    summary['iteration'] = int(current_iter)
    summary['manifest_path'] = manifest_path

    if save_scores:
        df.to_csv(os.path.join(iter_dir, 'train_scores.csv'), index=False)
    with open(os.path.join(iter_dir, 'summary.json'), 'w') as file:
        json.dump(_make_json_safe(summary), file, indent=2)

    plot_global_hist(df, os.path.join(iter_dir, 'hist_global.png'))
    plot_class_hists(df, iter_dir)
    plot_class_gap(df, os.path.join(iter_dir, 'class_gap.png'))

    if print_fn is not None:
        print_fn('warmup diagnosis iter {}: score_gap={}, train_noise_auroc={}, train_noise_ap={}'.format(
            current_iter,
            summary.get('score_gap'),
            summary.get('train_noise_auroc'),
            summary.get('train_noise_ap')))

    return {
        'summary': summary,
        'df': df,
        'iter_dir': iter_dir,
    }
