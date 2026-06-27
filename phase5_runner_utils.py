import json
import os

import numpy as np
import pandas as pd
import torch

from gbps_runner_utils import summarize_class_label_counts, summarize_contamination_labels
from warmup_diag import load_injected_manifest


DEFAULT_PHASE5_FLAG_THRESHOLDS = {
    'flag_wimg_threshold': 0.75,
    'flag_suppress_ratio_threshold': 0.25,
    'flag_min_wpatch_threshold': 0.5,
    'heavy_weight_threshold': 0.25,
}


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


def _safe_mean(values, default=None):
    if values is None:
        return default
    if isinstance(values, pd.Series):
        values = pd.to_numeric(values, errors='coerce')
        if len(values) == 0:
            return default
        value = values.mean()
    else:
        values = np.asarray(values)
        if values.size == 0:
            return default
        value = np.mean(values)
    if pd.isna(value):
        return default
    return float(value)


def _safe_min(values, default=None):
    if values is None:
        return default
    if isinstance(values, pd.Series):
        values = pd.to_numeric(values, errors='coerce')
        if len(values) == 0:
            return default
        value = values.min()
    else:
        values = np.asarray(values)
        if values.size == 0:
            return default
        value = np.min(values)
    if pd.isna(value):
        return default
    return float(value)


def _safe_max(values, default=None):
    if values is None:
        return default
    if isinstance(values, pd.Series):
        values = pd.to_numeric(values, errors='coerce')
        if len(values) == 0:
            return default
        value = values.max()
    else:
        values = np.asarray(values)
        if values.size == 0:
            return default
        value = np.max(values)
    if pd.isna(value):
        return default
    return float(value)


def _safe_ratio(numerator, denominator, default=None):
    if denominator in [0, None]:
        return default
    return float(numerator) / float(denominator)


def resolve_phase5_manifest_info(data_path, requested_manifest_path=None):
    repo_root = os.path.dirname(os.path.abspath(__file__))
    contaminated_paths, resolved_manifest_path = load_injected_manifest(
        data_path,
        repo_root,
        manifest_path=requested_manifest_path,
    )
    requested_path = None if requested_manifest_path is None else os.path.abspath(requested_manifest_path)
    return {
        'contaminated_paths': contaminated_paths,
        'diag_manifest_path': resolved_manifest_path,
        'diag_manifest_requested_path': requested_path,
        'diag_has_contamination_labels': bool(contaminated_paths is not None),
        'diag_contamination_label_mode': 'manifest' if contaminated_paths is not None else 'none',
    }


def _prepare_phase5_sample_analysis_df(summary_df, thresholds=None):
    thresholds = dict(DEFAULT_PHASE5_FLAG_THRESHOLDS if thresholds is None else thresholds)
    if summary_df is None:
        return pd.DataFrame()

    df = summary_df.copy()
    if len(df) == 0:
        for column in [
            'patch_h', 'patch_w', 'patch_count', 'fallback_reason', 'diag_has_contamination_labels',
            'is_flagged_by_wimg', 'is_flagged_by_suppress_ratio', 'is_flagged_by_min_wpatch',
            'is_flagged_any', 'is_heavy_flagged',
        ]:
            if column not in df.columns:
                df[column] = []
        return df

    if 'patch_h' not in df.columns:
        df['patch_h'] = np.nan
    if 'patch_w' not in df.columns:
        df['patch_w'] = np.nan
    if 'patch_count' not in df.columns:
        df['patch_count'] = pd.to_numeric(df['patch_h'], errors='coerce') * pd.to_numeric(df['patch_w'], errors='coerce')
    if 'fallback_reason' not in df.columns:
        df['fallback_reason'] = 'none'
    df['fallback_reason'] = df['fallback_reason'].fillna('none').astype(str)

    if 'is_contaminated' in df.columns:
        labels = pd.to_numeric(df['is_contaminated'], errors='coerce')
        has_labels = bool(labels.isin([0, 1]).any())
        df['is_contaminated'] = labels
    else:
        has_labels = False
        df['is_contaminated'] = np.nan
    df['diag_has_contamination_labels'] = int(has_labels)

    w_img = pd.to_numeric(df.get('w_img', pd.Series(np.ones(len(df)))), errors='coerce').fillna(1.0)
    suppress_ratio = pd.to_numeric(df.get('suppress_ratio', pd.Series(np.zeros(len(df)))), errors='coerce').fillna(0.0)
    min_w_patch = pd.to_numeric(df.get('min_w_patch', pd.Series(np.ones(len(df)))), errors='coerce').fillna(1.0)
    mean_w_patch = pd.to_numeric(df.get('mean_w_patch', pd.Series(np.ones(len(df)))), errors='coerce').fillna(1.0)

    df['is_flagged_by_wimg'] = (w_img <= float(thresholds['flag_wimg_threshold'])).astype(int)
    df['is_flagged_by_suppress_ratio'] = (suppress_ratio >= float(thresholds['flag_suppress_ratio_threshold'])).astype(int)
    df['is_flagged_by_min_wpatch'] = (min_w_patch <= float(thresholds['flag_min_wpatch_threshold'])).astype(int)
    df['is_flagged_any'] = (
        (df['is_flagged_by_wimg'] > 0)
        | (df['is_flagged_by_suppress_ratio'] > 0)
        | (df['is_flagged_by_min_wpatch'] > 0)
    ).astype(int)
    df['is_heavy_flagged'] = (
        (w_img <= float(thresholds['heavy_weight_threshold']))
        | (min_w_patch <= float(thresholds['heavy_weight_threshold']))
        | (mean_w_patch <= float(thresholds['heavy_weight_threshold']))
    ).astype(int)
    return df


def _build_subset_summary(row_prefix, subset_df):
    row = {
        '{}_num_samples'.format(row_prefix): int(len(subset_df)),
        '{}_mean_w_img'.format(row_prefix): _safe_mean(subset_df.get('w_img'), default=1.0),
        '{}_min_w_img'.format(row_prefix): _safe_min(subset_df.get('w_img'), default=1.0),
        '{}_mean_w_patch'.format(row_prefix): _safe_mean(subset_df.get('mean_w_patch'), default=1.0),
        '{}_min_w_patch'.format(row_prefix): _safe_min(subset_df.get('min_w_patch'), default=1.0),
        '{}_mean_suppress_ratio'.format(row_prefix): _safe_mean(subset_df.get('suppress_ratio'), default=0.0),
        '{}_flagged_count'.format(row_prefix): int(pd.to_numeric(subset_df.get('is_flagged_any'), errors='coerce').fillna(0).sum()) if len(subset_df) > 0 else 0,
        '{}_heavy_flagged_count'.format(row_prefix): int(pd.to_numeric(subset_df.get('is_heavy_flagged'), errors='coerce').fillna(0).sum()) if len(subset_df) > 0 else 0,
        '{}_fallback_count'.format(row_prefix): int((subset_df.get('fallback_reason', pd.Series(dtype=object)).fillna('none') != 'none').sum()) if len(subset_df) > 0 else 0,
    }
    row['{}_flagged_ratio'.format(row_prefix)] = _safe_ratio(row['{}_flagged_count'.format(row_prefix)], row['{}_num_samples'.format(row_prefix)], default=0.0)
    row['{}_heavy_flagged_ratio'.format(row_prefix)] = _safe_ratio(row['{}_heavy_flagged_count'.format(row_prefix)], row['{}_num_samples'.format(row_prefix)], default=0.0)
    return row


def _build_grouped_summary(df, group_columns):
    if df is None or len(df) == 0 or len(group_columns) == 0:
        return pd.DataFrame()

    existing_columns = [column for column in group_columns if column in df.columns]
    if len(existing_columns) == 0:
        return pd.DataFrame()

    rows = []
    grouped = df.groupby(existing_columns, sort=True, dropna=False)
    for group_keys, subset_df in grouped:
        if len(existing_columns) == 1:
            group_keys = (group_keys,)
        row = {column: value for column, value in zip(existing_columns, group_keys)}
        row['num_samples'] = int(len(subset_df))
        row['group_size_mean'] = _safe_mean(subset_df.get('group_size'))
        row['w_img_mean'] = _safe_mean(subset_df.get('w_img'), default=1.0)
        row['w_img_min'] = _safe_min(subset_df.get('w_img'), default=1.0)
        row['mean_w_patch_mean'] = _safe_mean(subset_df.get('mean_w_patch'), default=1.0)
        row['min_w_patch_min'] = _safe_min(subset_df.get('min_w_patch'), default=1.0)
        row['mean_D_patch_mean'] = _safe_mean(subset_df.get('mean_D_patch'), default=0.0)
        row['max_D_patch_max'] = _safe_max(subset_df.get('max_D_patch'), default=0.0)
        row['suppress_ratio_mean'] = _safe_mean(subset_df.get('suppress_ratio'), default=0.0)
        row['flagged_count'] = int(pd.to_numeric(subset_df.get('is_flagged_any'), errors='coerce').fillna(0).sum())
        row['flagged_ratio'] = _safe_ratio(row['flagged_count'], row['num_samples'], default=0.0)
        row['heavy_flagged_count'] = int(pd.to_numeric(subset_df.get('is_heavy_flagged'), errors='coerce').fillna(0).sum())
        row['heavy_flagged_ratio'] = _safe_ratio(row['heavy_flagged_count'], row['num_samples'], default=0.0)
        row['fallback_count'] = int((subset_df.get('fallback_reason', pd.Series(dtype=object)).fillna('none') != 'none').sum())
        row['fallback_ratio'] = _safe_ratio(row['fallback_count'], row['num_samples'], default=0.0)
        row.update(summarize_contamination_labels(subset_df))

        labels = pd.to_numeric(subset_df.get('is_contaminated'), errors='coerce') if 'is_contaminated' in subset_df.columns else pd.Series(dtype=float)
        clean_df = subset_df.loc[labels == 0]
        contaminated_df = subset_df.loc[labels == 1]
        row.update(_build_subset_summary('clean', clean_df))
        row.update(_build_subset_summary('contaminated', contaminated_df))
        rows.append(row)

    return pd.DataFrame(rows)


def _build_fallback_reason_summary(df):
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=['fallback_reason', 'num_samples'])

    fallback_series = df['fallback_reason'].fillna('none').astype(str)
    rows = []
    for fallback_reason, subset_df in df.groupby(fallback_series, sort=True):
        row = {'fallback_reason': str(fallback_reason), 'num_samples': int(len(subset_df))}
        row['sample_ratio'] = _safe_ratio(row['num_samples'], len(df), default=0.0)
        row['flagged_count'] = int(pd.to_numeric(subset_df.get('is_flagged_any'), errors='coerce').fillna(0).sum())
        row['flagged_ratio'] = _safe_ratio(row['flagged_count'], row['num_samples'], default=0.0)
        row.update(summarize_contamination_labels(subset_df))
        rows.append(row)
    return pd.DataFrame(rows)


def build_phase5_analysis_artifacts(summary_df, global_summary=None, thresholds=None):
    thresholds = dict(DEFAULT_PHASE5_FLAG_THRESHOLDS if thresholds is None else thresholds)
    sample_df = _prepare_phase5_sample_analysis_df(summary_df, thresholds=thresholds)
    class_df = _build_grouped_summary(sample_df, ['class_id', 'class_name'])
    group_df = _build_grouped_summary(sample_df, ['group_id'])
    class_group_df = _build_grouped_summary(sample_df, ['class_id', 'class_name', 'group_id'])
    fallback_reason_df = _build_fallback_reason_summary(sample_df)

    summary = {} if global_summary is None else dict(global_summary)
    summary.update(thresholds)
    summary.update(summarize_contamination_labels(sample_df))
    summary['class_sample_counts'] = summarize_class_label_counts(sample_df)
    summary['fallback_reason_counts'] = {
        str(reason): int(count)
        for reason, count in sample_df['fallback_reason'].fillna('none').astype(str).value_counts().sort_index().items()
    } if len(sample_df) > 0 else {}

    summary['mean_w_img'] = _safe_mean(sample_df.get('w_img'), default=1.0)
    summary['min_w_img'] = _safe_min(sample_df.get('w_img'), default=1.0)
    summary['mean_w_patch'] = _safe_mean(sample_df.get('mean_w_patch'), default=1.0)
    summary['min_w_patch'] = _safe_min(sample_df.get('min_w_patch'), default=1.0)
    summary['mean_suppress_ratio'] = _safe_mean(sample_df.get('suppress_ratio'), default=0.0)
    summary['fallback_sample_count'] = int((sample_df.get('fallback_reason', pd.Series(dtype=object)).fillna('none') != 'none').sum()) if len(sample_df) > 0 else 0
    summary['flagged_sample_count'] = int(pd.to_numeric(sample_df.get('is_flagged_any'), errors='coerce').fillna(0).sum()) if len(sample_df) > 0 else 0
    summary['heavy_flagged_sample_count'] = int(pd.to_numeric(sample_df.get('is_heavy_flagged'), errors='coerce').fillna(0).sum()) if len(sample_df) > 0 else 0

    labels = pd.to_numeric(sample_df.get('is_contaminated'), errors='coerce') if 'is_contaminated' in sample_df.columns else pd.Series(dtype=float)
    clean_df = sample_df.loc[labels == 0]
    contaminated_df = sample_df.loc[labels == 1]
    summary['mean_w_img_clean'] = _safe_mean(clean_df.get('w_img'))
    summary['mean_w_img_contaminated'] = _safe_mean(contaminated_df.get('w_img'))
    summary['mean_suppress_ratio_clean'] = _safe_mean(clean_df.get('suppress_ratio'))
    summary['mean_suppress_ratio_contaminated'] = _safe_mean(contaminated_df.get('suppress_ratio'))
    summary['true_abnormal_flagged_count'] = int(pd.to_numeric(contaminated_df.get('is_flagged_any'), errors='coerce').fillna(0).sum()) if len(contaminated_df) > 0 else 0
    summary['clean_flagged_count'] = int(pd.to_numeric(clean_df.get('is_flagged_any'), errors='coerce').fillna(0).sum()) if len(clean_df) > 0 else 0
    summary['true_abnormal_flagged_ratio'] = _safe_ratio(summary['true_abnormal_flagged_count'], len(contaminated_df), default=None)
    summary['clean_flagged_ratio'] = _safe_ratio(summary['clean_flagged_count'], len(clean_df), default=None)

    return {
        'sample_df': sample_df,
        'class_df': class_df,
        'group_df': group_df,
        'class_group_df': class_group_df,
        'fallback_reason_df': fallback_reason_df,
        'summary': summary,
    }


def save_phase5_artifacts(save_dir, summary_df, extra_summary=None, analysis_outputs=None):
    os.makedirs(save_dir, exist_ok=True)
    sample_df = summary_df
    summary_payload = extra_summary
    if analysis_outputs is not None:
        sample_df = analysis_outputs.get('sample_df', sample_df)
        summary_payload = analysis_outputs.get('summary', summary_payload)

    if sample_df is not None:
        sample_df.to_csv(os.path.join(save_dir, 'reliability_summary.csv'), index=False)
        sample_df.to_csv(os.path.join(save_dir, 'sample_reliability_analysis.csv'), index=False)
    if analysis_outputs is not None:
        table_map = {
            'class_df': 'class_reliability_summary.csv',
            'group_df': 'group_reliability_summary.csv',
            'class_group_df': 'class_group_reliability_summary.csv',
            'fallback_reason_df': 'fallback_reason_summary.csv',
        }
        for key, filename in table_map.items():
            table = analysis_outputs.get(key)
            if table is not None:
                table.to_csv(os.path.join(save_dir, filename), index=False)
    if summary_payload is not None:
        with open(os.path.join(save_dir, 'reliability_summary.json'), 'w') as file:
            json.dump(_make_json_safe(summary_payload), file, indent=2)
    os.makedirs(os.path.join(save_dir, 'vis'), exist_ok=True)


def save_phase5_bank(save_dir, reliability_bank, bank_save_path=None):
    os.makedirs(save_dir, exist_ok=True)
    bank_path = os.path.join(save_dir, 'reliability_bank.pt') if bank_save_path is None else os.path.abspath(bank_save_path)
    os.makedirs(os.path.dirname(bank_path), exist_ok=True)
    torch.save(reliability_bank, bank_path)
    return bank_path


def build_phase5_weight_summary_from_bank(reliability_bank):
    rows = []
    for sample_key in sorted(reliability_bank.keys()):
        entry = reliability_bank[sample_key]
        w_patch = entry['w_patch'].detach().cpu().float()
        rows.append({
            'sample_key': int(sample_key),
            'group_id': int(entry.get('group_id', -1)),
            'w_img': float(entry.get('w_img', 1.0)),
            'mean_w_patch': float(w_patch.mean().item()),
            'min_w_patch': float(w_patch.min().item()),
            'suppress_ratio': float((w_patch < 1.0).float().mean().item()),
            'fallback': bool(entry.get('fallback', False)),
        })
    columns = ['sample_key', 'group_id', 'w_img', 'mean_w_patch', 'min_w_patch', 'suppress_ratio', 'fallback']
    return pd.DataFrame(rows, columns=columns)


def save_phase5_eval_outputs(run_dir, final_eval_summary, eval_history_rows=None):
    os.makedirs(run_dir, exist_ok=True)
    summary_path = os.path.join(run_dir, 'final_eval_summary.json')
    with open(summary_path, 'w') as file:
        json.dump(_make_json_safe(final_eval_summary), file, indent=2)
    if eval_history_rows is not None:
        pd.DataFrame(eval_history_rows).to_csv(os.path.join(run_dir, 'eval_history.csv'), index=False)
    return summary_path


def save_phase5_checkpoint(run_dir, model, iteration, args, extra_state=None):
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_path = os.path.join(run_dir, 'phase5_final_checkpoint.pt')
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'iteration': int(iteration),
        'args': _make_json_safe(vars(args)),
    }
    if extra_state is not None:
        checkpoint.update(extra_state)
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path
