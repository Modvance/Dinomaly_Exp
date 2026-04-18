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
    if has_contamination:
        clean_mask = df['is_contaminated'] == 0
        noisy_mask = df['is_contaminated'] == 1
        summary.update({
            'num_clean': int(clean_mask.sum()),
            'num_noisy': int(noisy_mask.sum()),
            'clean_mean_score': _safe_float(_safe_mean(df.loc[clean_mask, 'image_score'])),
            'noisy_mean_score': _safe_float(_safe_mean(df.loc[noisy_mask, 'image_score'])),
            'score_gap': _safe_float(_safe_mean(df.loc[noisy_mask, 'image_score']) - _safe_mean(df.loc[clean_mask, 'image_score'])),
            'train_noise_auroc': _safe_float(_safe_binary_metric(roc_auc_score, df['is_contaminated'], df['image_score'])),
            'train_noise_ap': _safe_float(_safe_binary_metric(average_precision_score, df['is_contaminated'], df['image_score'])),
        })
    else:
        summary.update({
            'num_clean': None,
            'num_noisy': None,
            'clean_mean_score': None,
            'noisy_mean_score': None,
            'score_gap': None,
            'train_noise_auroc': None,
            'train_noise_ap': None,
        })

    bucket_metrics = {}
    if 'tail_bucket' in df.columns and (df['tail_bucket'].fillna('') != '').any():
        bucket_df_source = df[df['tail_bucket'].fillna('') != '']
        for bucket, bucket_df in bucket_df_source.groupby('tail_bucket'):
            bucket_metrics[str(bucket)] = {
                'num_samples': int(len(bucket_df)),
                'train_noise_auroc': _safe_float(_safe_binary_metric(roc_auc_score, bucket_df['is_contaminated'], bucket_df['image_score'])) if 'is_contaminated' in bucket_df.columns else None,
                'train_noise_ap': _safe_float(_safe_binary_metric(average_precision_score, bucket_df['is_contaminated'], bucket_df['image_score'])) if 'is_contaminated' in bucket_df.columns else None,
                'clean_mean_score': _safe_float(_safe_mean(bucket_df.loc[bucket_df['is_contaminated'] == 0, 'image_score'])) if 'is_contaminated' in bucket_df.columns else None,
                'noisy_mean_score': _safe_float(_safe_mean(bucket_df.loc[bucket_df['is_contaminated'] == 1, 'image_score'])) if 'is_contaminated' in bucket_df.columns else None,
            }
    summary['bucket_metrics'] = bucket_metrics

    class_metrics = {}
    if 'class_name' in df.columns:
        for class_name, class_df in df.groupby('class_name'):
            clean_mean = _safe_mean(class_df.loc[class_df['is_contaminated'] == 0, 'image_score']) if 'is_contaminated' in class_df.columns else float('nan')
            noisy_mean = _safe_mean(class_df.loc[class_df['is_contaminated'] == 1, 'image_score']) if 'is_contaminated' in class_df.columns else float('nan')
            class_metrics[str(class_name)] = {
                'num_samples': int(len(class_df)),
                'clean_mean_score': _safe_float(clean_mean),
                'noisy_mean_score': _safe_float(noisy_mean),
                'score_gap': _safe_float(noisy_mean - clean_mean),
                'train_noise_auroc': _safe_float(_safe_binary_metric(roc_auc_score, class_df['is_contaminated'], class_df['image_score'])) if 'is_contaminated' in class_df.columns else None,
            }
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


def plot_bucket_hist(df, save_path):
    if 'tail_bucket' not in df.columns or not (df['tail_bucket'].fillna('') != '').any():
        return False

    buckets = sorted(df.loc[df['tail_bucket'].fillna('') != '', 'tail_bucket'].unique())
    if len(buckets) == 0:
        return False

    fig, axes = plt.subplots(len(buckets), 1, figsize=(8, max(4, 3 * len(buckets))), squeeze=False)
    for axis, bucket in zip(axes.flatten(), buckets):
        bucket_df = df[df['tail_bucket'] == bucket]
        if 'is_contaminated' in bucket_df.columns and bucket_df['is_contaminated'].notna().any():
            clean_scores = bucket_df.loc[bucket_df['is_contaminated'] == 0, 'image_score']
            noisy_scores = bucket_df.loc[bucket_df['is_contaminated'] == 1, 'image_score']
            if len(clean_scores) > 0:
                axis.hist(clean_scores, bins=20, alpha=0.6, label='clean')
            if len(noisy_scores) > 0:
                axis.hist(noisy_scores, bins=20, alpha=0.6, label='noisy')
            axis.legend()
        else:
            axis.hist(bucket_df['image_score'], bins=20, alpha=0.8)
        axis.set_title(str(bucket))
        axis.set_xlabel('image_score')
        axis.set_ylabel('count')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
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
    plot_bucket_hist(df, os.path.join(iter_dir, 'hist_buckets.png'))
    plot_class_gap(df, os.path.join(iter_dir, 'class_gap.png'))

    if print_fn is not None:
        print_fn('warmup diagnosis iter {}: score_gap={}, train_noise_auroc={}, train_noise_ap={}'.format(
            current_iter,
            summary.get('score_gap'),
            summary.get('train_noise_auroc'),
            summary.get('train_noise_ap')))

    return summary
