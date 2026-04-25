import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

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


def _resolve_manifest_candidates(data_path, repo_root):
    data_parts = Path(os.path.abspath(data_path)).parts
    manifest_root = Path(repo_root) / 'manifest' / 'mvtecad-nlt'
    candidates = []

    for index, part in enumerate(data_parts):
        if part.startswith('step_k') or part == 'pareto':
            if index + 1 < len(data_parts) and data_parts[index + 1].startswith('seed'):
                candidates.append(manifest_root / part / data_parts[index + 1] / 'inject_defects.txt')

    if os.path.isfile(data_path):
        candidates.append(Path(data_path))

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


def run_one_warmup_diagnosis(model, loader, device, save_dir, current_iter, print_fn=None, max_ratio=0.01,
                             resize_mask=256, manifest_path=None):
    iter_dir = os.path.join(save_dir, 'iter_{:05d}'.format(current_iter))
    os.makedirs(iter_dir, exist_ok=True)

    df = score_trainset(model, loader, device, max_ratio=max_ratio, resize_mask=resize_mask)
    summary = compute_warmup_diagnostics(df)
    summary['iteration'] = int(current_iter)
    summary['manifest_path'] = manifest_path

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

    return summary
