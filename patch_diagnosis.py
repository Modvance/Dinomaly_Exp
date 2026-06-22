import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture


def _safe_float(value):
    if value is None:
        return None
    if isinstance(value, (np.floating, float)) and (np.isnan(value) or np.isinf(value)):
        return None
    return float(value)


def parse_patch_grid_size(patch_grid_size):
    if isinstance(patch_grid_size, str):
        parts = [part.strip() for part in patch_grid_size.split(',') if part.strip()]
        if len(parts) == 1:
            size = int(parts[0])
            return size, size
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
        raise ValueError('invalid patch_grid_size: {}'.format(patch_grid_size))
    if isinstance(patch_grid_size, (tuple, list)):
        if len(patch_grid_size) == 1:
            size = int(patch_grid_size[0])
            return size, size
        if len(patch_grid_size) == 2:
            return int(patch_grid_size[0]), int(patch_grid_size[1])
        raise ValueError('invalid patch_grid_size: {}'.format(patch_grid_size))
    size = int(patch_grid_size)
    return size, size


def build_patch_score_grid(anomaly_maps, patch_grid_size):
    patch_h, patch_w = parse_patch_grid_size(patch_grid_size)
    if anomaly_maps.ndim == 3:
        anomaly_maps = anomaly_maps.unsqueeze(1)
    if anomaly_maps.ndim != 4:
        raise ValueError('anomaly_maps must have shape [B, H, W] or [B, 1, H, W]')
    patch_scores = F.interpolate(anomaly_maps.float(), size=(patch_h, patch_w), mode='area')
    return patch_scores[:, 0]


def robust_normalize_patch_scores(scores, eps=1e-6):
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if scores.size == 0:
        raise ValueError('cannot normalize empty patch scores')
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median))) + float(eps)
    z_scores = (scores - median) / mad
    return z_scores.astype(float), {'score_median': median, 'score_mad': mad}


def fit_patch_gmms(z_scores, eps=1e-6, random_state=0):
    z_scores = np.asarray(z_scores, dtype=float).reshape(-1)
    if z_scores.size == 0:
        raise ValueError('cannot fit GMM on empty patch scores')

    x = z_scores.reshape(-1, 1)
    gmm1 = GaussianMixture(n_components=1, covariance_type='full', reg_covar=eps, random_state=random_state)
    gmm1.fit(x)
    bic_1 = float(gmm1.bic(x))

    if z_scores.size < 2 or np.allclose(z_scores, z_scores[0]):
        mean_value = float(np.mean(z_scores))
        sigma_value = float(np.std(z_scores) + eps)
        posterior_high = np.full(z_scores.shape[0], 0.5, dtype=float)
        return {
            'bic_1': bic_1,
            'bic_2': bic_1,
            'delta_bic': 0.0,
            'mu_low': mean_value,
            'mu_high': mean_value,
            'sigma_low': sigma_value,
            'sigma_high': sigma_value,
            'pi_low': 0.5,
            'pi_high': 0.5,
            'posterior_high': posterior_high,
        }

    try:
        gmm2 = GaussianMixture(n_components=2, covariance_type='full', reg_covar=eps, random_state=random_state)
        gmm2.fit(x)
        bic_2 = float(gmm2.bic(x))
        means = gmm2.means_.reshape(-1)
        variances = gmm2.covariances_.reshape(2, -1)[:, 0]
        sigmas = np.sqrt(np.maximum(variances, eps))
        weights = gmm2.weights_.reshape(-1)
        order = np.argsort(means)
        low_idx, high_idx = int(order[0]), int(order[1])
        posterior_high = gmm2.predict_proba(x)[:, high_idx]
        return {
            'bic_1': bic_1,
            'bic_2': bic_2,
            'delta_bic': float(bic_1 - bic_2),
            'mu_low': float(means[low_idx]),
            'mu_high': float(means[high_idx]),
            'sigma_low': float(sigmas[low_idx]),
            'sigma_high': float(sigmas[high_idx]),
            'pi_low': float(weights[low_idx]),
            'pi_high': float(weights[high_idx]),
            'posterior_high': posterior_high.astype(float),
        }
    except Exception:
        mean_value = float(np.mean(z_scores))
        sigma_value = float(np.std(z_scores) + eps)
        posterior_high = np.full(z_scores.shape[0], 0.5, dtype=float)
        return {
            'bic_1': bic_1,
            'bic_2': bic_1,
            'delta_bic': 0.0,
            'mu_low': mean_value,
            'mu_high': mean_value,
            'sigma_low': sigma_value,
            'sigma_high': sigma_value,
            'pi_low': 0.5,
            'pi_high': 0.5,
            'posterior_high': posterior_high,
        }


def compute_patch_sep_conf(gmm_stats, eps=1e-6):
    sep = float((gmm_stats['mu_high'] - gmm_stats['mu_low']) / max(gmm_stats['sigma_low'] + gmm_stats['sigma_high'], eps))
    posterior_high = np.asarray(gmm_stats['posterior_high'], dtype=float)
    posterior = np.stack([1.0 - posterior_high, posterior_high], axis=1)
    entropy = -np.sum(posterior * np.log(posterior + float(eps)), axis=1)
    conf = float(np.clip(1.0 - float(np.mean(entropy) / math.log(2.0)), 0.0, 1.0)) if posterior.shape[0] > 0 else 0.0
    return sep, conf


def evaluate_patch_gate(delta_bic, sep, conf, pi_high, args):
    is_active = (
        float(delta_bic) > float(args.patch_tau_bic)
        and float(sep) > float(args.patch_tau_sep)
        and float(conf) > float(args.patch_tau_conf)
        and float(args.patch_pi_min) <= float(pi_high) <= float(args.patch_pi_max)
    )
    return bool(is_active), ('active' if is_active else 'gate_rejected')


def _apply_patch_max_suppress_ratio(weight_grid, p_high_grid, max_suppress_ratio):
    weight_grid = np.asarray(weight_grid, dtype=float).copy()
    p_high_grid = np.asarray(p_high_grid, dtype=float).copy()
    suppressed_mask = weight_grid < 1.0
    max_suppress_count = int(math.floor(weight_grid.size * float(max_suppress_ratio)))
    if suppressed_mask.sum() <= max_suppress_count:
        return weight_grid
    if max_suppress_count <= 0:
        return np.ones_like(weight_grid, dtype=float)

    suppressed_indices = np.where(suppressed_mask.reshape(-1))[0]
    suppressed_scores = p_high_grid.reshape(-1)[suppressed_indices]
    sorted_order = np.argsort(-suppressed_scores)
    keep_indices = suppressed_indices[sorted_order[:max_suppress_count]]

    adjusted = np.ones_like(weight_grid.reshape(-1), dtype=float)
    original = weight_grid.reshape(-1)
    adjusted[keep_indices] = original[keep_indices]
    return adjusted.reshape(weight_grid.shape)


def compute_patch_weight_grid(p_high_grid, active, args):
    p_high_grid = np.asarray(p_high_grid, dtype=float)
    if not active:
        return np.ones_like(p_high_grid, dtype=float)

    mode = getattr(args, 'patch_weight_mode', 'hybrid')
    tau_soft = float(getattr(args, 'patch_tau_soft', 0.5))
    tau_hard = float(getattr(args, 'patch_tau_hard', 0.95))
    w_min = float(getattr(args, 'patch_w_min', 0.05))
    patch_lambda = float(getattr(args, 'patch_lambda', 0.95))

    if mode == 'hard':
        weight_grid = np.where(p_high_grid > tau_hard, 0.0, 1.0)
    elif mode == 'soft':
        weight_grid = np.maximum(w_min, 1.0 - patch_lambda * p_high_grid)
    elif mode == 'hybrid':
        weight_grid = np.ones_like(p_high_grid, dtype=float)
        hard_mask = p_high_grid > tau_hard
        soft_mask = (p_high_grid > tau_soft) & (~hard_mask)
        weight_grid[hard_mask] = 0.0
        weight_grid[soft_mask] = np.maximum(w_min, 1.0 - patch_lambda * p_high_grid[soft_mask])
    else:
        raise ValueError('unsupported patch_weight_mode: {}'.format(mode))

    weight_grid = _apply_patch_max_suppress_ratio(
        weight_grid,
        p_high_grid,
        max_suppress_ratio=float(getattr(args, 'patch_max_suppress_ratio', 0.30)),
    )
    return weight_grid.astype(float)


def compute_patch_diagnosis_for_image(patch_scores, args, random_state=0):
    patch_scores = np.asarray(patch_scores, dtype=float)
    patch_h, patch_w = patch_scores.shape
    flat_scores = patch_scores.reshape(-1)
    z_scores, robust_stats = robust_normalize_patch_scores(flat_scores)

    summary = {
        'active': False,
        'skip_reason': 'gate_rejected',
        'delta_bic': None,
        'sep': None,
        'conf': None,
        'pi_high': None,
        'E_i': 0.0,
        'mean_patch_score': float(np.mean(flat_scores)),
        'max_patch_score': float(np.max(flat_scores)),
        'score_median': float(robust_stats['score_median']),
        'score_mad': float(robust_stats['score_mad']),
        'patch_h': int(patch_h),
        'patch_w': int(patch_w),
    }

    if float(robust_stats['score_mad']) < float(getattr(args, 'patch_min_mad', 1e-6)):
        p_high_grid = np.zeros_like(patch_scores, dtype=float)
        weight_grid = np.ones_like(patch_scores, dtype=float)
        summary.update({
            'skip_reason': 'low_mad',
            'mean_weight': float(np.mean(weight_grid)),
            'min_weight': float(np.min(weight_grid)),
            'suppress_ratio': 0.0,
        })
        return summary, {
            'patch_scores': torch.tensor(patch_scores, dtype=torch.float32),
            'z_scores': torch.tensor(z_scores.reshape(patch_h, patch_w), dtype=torch.float32),
            'p_high': torch.tensor(p_high_grid, dtype=torch.float32),
            'weight': torch.tensor(weight_grid, dtype=torch.float32),
        }

    gmm_stats = fit_patch_gmms(z_scores, eps=1e-6, random_state=random_state)
    sep, conf = compute_patch_sep_conf(gmm_stats)
    is_active, skip_reason = evaluate_patch_gate(
        gmm_stats['delta_bic'],
        sep,
        conf,
        gmm_stats['pi_high'],
        args,
    )
    E_i = float(sep) * float(conf) if is_active else 0.0
    p_high_grid = np.asarray(gmm_stats['posterior_high'], dtype=float).reshape(patch_h, patch_w)
    weight_grid = compute_patch_weight_grid(p_high_grid, is_active, args)
    suppress_ratio = float(np.mean(weight_grid.reshape(-1) < 1.0))

    summary.update({
        'active': bool(is_active),
        'skip_reason': skip_reason,
        'delta_bic': _safe_float(gmm_stats['delta_bic']),
        'sep': _safe_float(sep),
        'conf': _safe_float(conf),
        'pi_high': _safe_float(gmm_stats['pi_high']),
        'E_i': _safe_float(E_i),
        'mean_weight': float(np.mean(weight_grid)),
        'min_weight': float(np.min(weight_grid)),
        'suppress_ratio': suppress_ratio,
    })
    return summary, {
        'patch_scores': torch.tensor(patch_scores, dtype=torch.float32),
        'z_scores': torch.tensor(z_scores.reshape(patch_h, patch_w), dtype=torch.float32),
        'p_high': torch.tensor(p_high_grid, dtype=torch.float32),
        'weight': torch.tensor(weight_grid, dtype=torch.float32),
    }


def build_patch_diagnosis_table(scored_df, patch_grids, args, random_state=0):
    if len(scored_df) != int(patch_grids.shape[0]):
        raise ValueError('scored_df and patch_grids length mismatch')

    rows = []
    detail_by_sample_key = {}
    for index, row in scored_df.reset_index(drop=True).iterrows():
        sample_key = int(row['sample_idx'])
        patch_scores = patch_grids[index].detach().cpu().numpy()
        summary, detail = compute_patch_diagnosis_for_image(
            patch_scores,
            args=args,
            random_state=random_state + index,
        )
        diagnosis_row = {
            'sample_key': int(sample_key),
            'sample_idx': int(sample_key),
            'img_path': row['img_path'],
            'image_score': float(row['image_score']),
            'active': bool(summary['active']),
            'skip_reason': summary['skip_reason'],
            'delta_bic': summary['delta_bic'],
            'sep': summary['sep'],
            'conf': summary['conf'],
            'pi_high': summary['pi_high'],
            'E_i': float(summary['E_i']),
            'mean_patch_score': float(summary['mean_patch_score']),
            'max_patch_score': float(summary['max_patch_score']),
            'mean_weight': float(summary['mean_weight']),
            'min_weight': float(summary['min_weight']),
            'suppress_ratio': float(summary['suppress_ratio']),
            'patch_h': int(summary['patch_h']),
            'patch_w': int(summary['patch_w']),
            'score_median': float(summary['score_median']),
            'score_mad': float(summary['score_mad']),
        }
        for optional_column in ['class_id', 'class_name', 'base_idx', 'is_contaminated']:
            if optional_column in row.index:
                diagnosis_row[optional_column] = row[optional_column]
        rows.append(diagnosis_row)
        detail_by_sample_key[int(sample_key)] = {
            'patch_scores': detail['patch_scores'],
            'z_scores': detail['z_scores'],
            'p_high': detail['p_high'],
            'weight': detail['weight'],
            'active': bool(summary['active']),
            'skip_reason': summary['skip_reason'],
        }

    diagnosis_df = pd.DataFrame(rows)
    payload = {
        'by_sample_key': detail_by_sample_key,
        'patch_grid_size': [int(patch_grids.shape[1]), int(patch_grids.shape[2])],
    }
    return diagnosis_df, payload


def compute_patch_u_with_bootstrap(diagnosis_df, B, args):
    if len(diagnosis_df) == 0:
        raise ValueError('patch diagnosis is empty')

    E_values = diagnosis_df['E_i'].to_numpy(dtype=float)
    active_mask = diagnosis_df['active'].to_numpy(dtype=bool)
    patch_U = float(np.mean(E_values))
    active_count = int(active_mask.sum())
    total_count = int(len(diagnosis_df))
    active_ratio = float(active_count / total_count) if total_count > 0 else 0.0
    mean_active_E = float(np.mean(E_values[active_mask])) if active_count > 0 else 0.0
    mean_all_E = float(np.mean(E_values)) if total_count > 0 else 0.0

    bootstrap_rows = []
    bootstrap_values = []
    bootstrap_B = max(1, int(B))
    random_state = np.random.RandomState(0)
    for bootstrap_id in range(bootstrap_B):
        sampled_indices = random_state.randint(0, total_count, size=total_count)
        sampled_E = E_values[sampled_indices]
        U_b = float(np.mean(sampled_E)) if len(sampled_E) > 0 else 0.0
        sampled_active = active_mask[sampled_indices]
        bootstrap_rows.append({
            'bootstrap_id': int(bootstrap_id),
            'U_b': U_b,
            'active_ratio_b': float(np.mean(sampled_active)) if len(sampled_active) > 0 else 0.0,
            'active_count_b': int(sampled_active.sum()),
        })
        bootstrap_values.append(U_b)

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    patch_se = float(np.std(np.asarray(bootstrap_values, dtype=float), ddof=0)) if len(bootstrap_values) > 0 else 0.0
    patch_u_mean = float(np.mean(np.asarray(bootstrap_values, dtype=float))) if len(bootstrap_values) > 0 else patch_U
    summary = {
        'patch_U': patch_U,
        'patch_SE': patch_se,
        'patch_U_bootstrap_mean': patch_u_mean,
        'patch_active_ratio': active_ratio,
        'patch_active_count': active_count,
        'patch_total_count': total_count,
        'patch_mean_active_E': mean_active_E,
        'patch_mean_all_E': mean_all_E,
        'patch_min_active_ratio': float(getattr(args, 'patch_min_active_ratio', 0.01)),
    }
    return {
        'patch_U': patch_U,
        'patch_SE': patch_se,
        'bootstrap_df': bootstrap_df,
        'bootstrap_values': bootstrap_values,
        'summary': summary,
    }


def build_patch_weight_bank(diagnosis_df, payload, args, force_all_ones=False):
    if 'by_sample_key' not in payload:
        raise ValueError('payload must contain by_sample_key')

    bank = {}
    summary_rows = []
    for _, row in diagnosis_df.iterrows():
        sample_key = int(row['sample_key'])
        detail = payload['by_sample_key'][sample_key]
        patch_scores = detail['patch_scores']
        p_high = detail['p_high']
        active = bool(row['active']) and not force_all_ones
        if force_all_ones:
            weight = torch.ones_like(p_high)
        else:
            weight = detail.get('weight')
            if weight is None:
                weight = torch.tensor(
                    compute_patch_weight_grid(p_high.detach().cpu().numpy(), active, args),
                    dtype=torch.float32,
                )
        bank[sample_key] = {
            'weight': weight.detach().cpu(),
            'p_high': p_high.detach().cpu(),
            'patch_scores': patch_scores.detach().cpu(),
            'active': bool(active),
            'summary': {
                'sample_key': int(sample_key),
                'img_path': row['img_path'],
                'skip_reason': 'skip_patch_denoise' if force_all_ones else row['skip_reason'],
                'E_i': float(row['E_i']),
            },
        }
        summary_row = dict(row)
        summary_row['active'] = bool(active)
        summary_row['skip_reason'] = 'skip_patch_denoise' if force_all_ones else row['skip_reason']
        summary_row['mean_weight'] = float(weight.mean().item())
        summary_row['min_weight'] = float(weight.min().item())
        summary_row['suppress_ratio'] = float((weight < 1.0).float().mean().item())
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    return bank, summary_df
