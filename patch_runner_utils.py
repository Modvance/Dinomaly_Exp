import json
import os

import numpy as np
import pandas as pd
import torch


def _safe_float(value):
    if value is None:
        return None
    if isinstance(value, (np.floating, float)) and (np.isnan(value) or np.isinf(value)):
        return None
    return float(value)


def _make_json_safe(value):
    if isinstance(value, dict):
        return {key: _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_make_json_safe(item) for item in value]
    if torch.is_tensor(value):
        return _make_json_safe(value.detach().cpu().tolist())
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def save_patch_artifacts(iter_dir, diagnosis_df, bootstrap_df=None, extra_summary=None):
    os.makedirs(iter_dir, exist_ok=True)
    summary_path = os.path.join(iter_dir, 'patch_global_summary.json')
    summary = {}
    if os.path.isfile(summary_path):
        with open(summary_path) as file:
            summary = json.load(file)
    if extra_summary is not None:
        summary.update(_make_json_safe(extra_summary))
    with open(summary_path, 'w') as file:
        json.dump(_make_json_safe(summary), file, indent=2)

    if diagnosis_df is not None:
        diagnosis_df.to_csv(os.path.join(iter_dir, 'patch_summary.csv'), index=False)
    if bootstrap_df is not None:
        bootstrap_df.to_csv(os.path.join(iter_dir, 'patch_bootstrap.csv'), index=False)

    os.makedirs(os.path.join(iter_dir, 'patch_vis'), exist_ok=True)


def save_best_patch_checkpoint(save_dir, model, current_iter):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, 'best_patch_warmup.ckpt')
    torch.save({
        'iteration': int(current_iter),
        'model_state_dict': model.state_dict(),
    }, checkpoint_path)
    return checkpoint_path


def save_patch_weight_artifacts(save_dir, bank, summary_df, extra_summary=None, bank_save_path=None):
    os.makedirs(save_dir, exist_ok=True)
    bank_path = os.path.join(save_dir, 'patch_weight_bank.pt') if bank_save_path is None else os.path.abspath(bank_save_path)
    os.makedirs(os.path.dirname(bank_path), exist_ok=True)
    torch.save(bank, bank_path)

    summary_path = os.path.join(save_dir, 'patch_weight_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    if extra_summary is not None:
        with open(os.path.join(save_dir, 'patch_weight_summary.json'), 'w') as file:
            json.dump(_make_json_safe(extra_summary), file, indent=2)

    os.makedirs(os.path.join(save_dir, 'patch_weight_vis'), exist_ok=True)
    return bank_path, summary_path


def evaluate_patch_ci_peak_trigger(U_t, SE_t, active_ratio_t, best_U, best_SE, best_iter,
                                   best_active_ratio, best_check_count, check_count, current_iter,
                                   ci_z, improve_eps, min_checks_before_trigger,
                                   min_checks_after_best, min_active_ratio, force_trigger=False):
    U_t = float(U_t)
    SE_t = float(SE_t)
    active_ratio_t = float(active_ratio_t)
    check_count = int(check_count)
    current_iter = int(current_iter)
    ci_z = float(ci_z)
    improve_eps = float(improve_eps)
    min_checks_before_trigger = int(min_checks_before_trigger)
    min_checks_after_best = int(min_checks_after_best)
    min_active_ratio = float(min_active_ratio)

    previous_best_U = None if best_U is None else float(best_U)
    previous_best_SE = None if best_SE is None else float(best_SE)
    previous_best_iter = None if best_iter is None else int(best_iter)
    previous_best_active_ratio = None if best_active_ratio is None else float(best_active_ratio)
    previous_best_check_count = None if best_check_count is None else int(best_check_count)

    improved = previous_best_U is None or U_t > (previous_best_U + improve_eps)
    if improved:
        best_U = U_t
        best_SE = SE_t
        best_iter = current_iter
        best_active_ratio = active_ratio_t
        best_check_count = check_count
    else:
        best_U = previous_best_U
        best_SE = previous_best_SE
        best_iter = previous_best_iter
        best_active_ratio = previous_best_active_ratio
        best_check_count = previous_best_check_count

    current_upper = float(U_t + ci_z * SE_t)
    best_lower = None if best_U is None or best_SE is None else float(best_U - ci_z * best_SE)
    checks_after_best = 0 if best_check_count is None else max(0, check_count - int(best_check_count))
    enough_checks_before_trigger = check_count >= min_checks_before_trigger
    enough_checks_after_best = checks_after_best >= min_checks_after_best
    ci_peak_ready = (
        previous_best_U is not None
        and best_lower is not None
        and enough_checks_before_trigger
        and enough_checks_after_best
        and current_upper < best_lower
    )
    has_noise_evidence = best_active_ratio is not None and float(best_active_ratio) >= min_active_ratio

    if force_trigger:
        if best_iter is not None and has_noise_evidence:
            selected_iter = int(best_iter)
            status = 'patch_forced_best'
        else:
            selected_iter = int(best_iter) if best_iter is not None else None
            status = 'patch_no_noise_forced'
    elif ci_peak_ready and best_iter is not None and has_noise_evidence:
        selected_iter = int(best_iter)
        status = 'patch_ci_peak'
    else:
        selected_iter = None
        status = 'none'

    summary = {
        'patch_U': U_t,
        'patch_SE': SE_t,
        'patch_active_ratio': active_ratio_t,
        'patch_best_U': _safe_float(best_U),
        'patch_best_SE': _safe_float(best_SE),
        'patch_best_iter': None if best_iter is None else int(best_iter),
        'patch_best_active_ratio': _safe_float(best_active_ratio),
        'patch_check_count': int(check_count),
        'patch_best_check_count': None if best_check_count is None else int(best_check_count),
        'patch_checks_after_best': int(checks_after_best),
        'patch_current_upper': _safe_float(current_upper),
        'patch_best_lower': _safe_float(best_lower),
        'patch_ci_z': ci_z,
        'patch_improve_eps': improve_eps,
        'patch_min_checks_before_trigger': min_checks_before_trigger,
        'patch_min_checks_after_best': min_checks_after_best,
        'patch_min_active_ratio': min_active_ratio,
        'patch_has_noise_evidence': bool(has_noise_evidence),
        'patch_improved': bool(improved),
        'patch_ci_peak_ready': bool(ci_peak_ready),
        'patch_status': status,
        'patch_triggered': bool(status != 'none'),
        'patch_selected_iter': None if selected_iter is None else int(selected_iter),
        'skip_patch_denoise': bool(status == 'patch_no_noise_forced' or not has_noise_evidence),
        'force_triggered': bool(force_trigger),
    }
    return {
        'status': status,
        'triggered': bool(status != 'none'),
        'selected_iter': selected_iter,
        'best_U': best_U,
        'best_SE': best_SE,
        'best_iter': best_iter,
        'best_active_ratio': best_active_ratio,
        'best_check_count': best_check_count,
        'summary': summary,
    }


def build_patch_weight_summary_from_bank(bank):
    rows = []
    for sample_key in sorted(bank.keys()):
        entry = bank[sample_key]
        weight = entry['weight'].detach().cpu().float()
        summary = dict(entry.get('summary', {}))
        summary.update({
            'sample_key': int(sample_key),
            'active': bool(entry.get('active', False)),
            'mean_weight': float(weight.mean().item()),
            'min_weight': float(weight.min().item()),
            'suppress_ratio': float((weight < 1.0).float().mean().item()),
        })
        rows.append(summary)
    return pd.DataFrame(rows)


def load_saved_patch_diagnosis(save_dir, iteration):
    iter_dir = os.path.join(save_dir, 'iter_{:05d}'.format(int(iteration)))
    diagnosis_path = os.path.join(iter_dir, 'patch_summary.csv')
    if not os.path.isfile(diagnosis_path):
        raise FileNotFoundError('saved patch diagnosis not found for iter {}: {}'.format(int(iteration), diagnosis_path))
    return pd.read_csv(diagnosis_path), diagnosis_path


def resolve_best_patch_diagnosis(current_diagnosis_df, current_iter, selected_iter, save_dir):
    if selected_iter is None:
        return None, None, None
    if int(selected_iter) == int(current_iter):
        return current_diagnosis_df.copy(), int(current_iter), None
    diagnosis_df, diagnosis_path = load_saved_patch_diagnosis(save_dir, selected_iter)
    return diagnosis_df.copy(), int(selected_iter), diagnosis_path
