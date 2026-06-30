import json
import math
import os

import numpy as np
import pandas as pd


PHASE6_AUG_TYPES = (
    'hflip',
    'rotate_small',
    'brightness_contrast',
    'translate_small',
    'resized_crop_weak',
)


TAIL_AUG_COLUMNS = [
    'aug_sample_key',
    'parent_sample_key',
    'parent_img_path',
    'group_id',
    'pseudo_group_id',
    'class_id',
    'base_idx',
    'sample_idx',
    'aug_seed',
    'aug_type',
    'aug_rank',
    'origin_score',
    'aug_score',
    'aug_score_std',
    'selected',
    'is_tail_reliable',
    'sampler_weight',
    'loss_weight',
]


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


def compute_group_headness(group_sizes, eps=1e-8):
    if isinstance(group_sizes, dict):
        group_sizes = pd.Series(group_sizes, dtype=np.float64)
    else:
        group_sizes = pd.Series(group_sizes).astype(np.float64)
    if len(group_sizes) == 0:
        return {}

    log_sizes = np.log(group_sizes + 1.0)
    log_min = float(log_sizes.min())
    log_max = float(log_sizes.max())
    denom = log_max - log_min
    if abs(denom) <= eps:
        return {group_id: 0.0 for group_id in group_sizes.index.tolist()}

    rho = ((log_sizes - log_min) / (denom + eps)).clip(0.0, 1.0)
    return {group_id: float(value) for group_id, value in rho.items()}


def _sort_kept_samples(kept_samples_df):
    if len(kept_samples_df) == 0:
        return kept_samples_df.copy().reset_index(drop=True)

    sort_columns = [column for column in ['sample_idx', 'class_id', 'base_idx', 'img_path'] if column in kept_samples_df.columns]
    if len(sort_columns) == 0:
        return kept_samples_df.copy().reset_index(drop=True)
    return kept_samples_df.sort_values(sort_columns).reset_index(drop=True)


def build_longtail_group_summary(kept_samples_df, group_col='group_id', alpha=0.5,
                                 epoch_num_samples=None, prune_group_counts=None,
                                 tail_rho_thr=0.3, head_rho_thr=0.7):
    if group_col not in kept_samples_df.columns:
        raise ValueError('long-tail group column not found: {}'.format(group_col))
    if len(kept_samples_df) == 0:
        raise ValueError('kept_samples_df is empty')

    group_sizes = kept_samples_df.groupby(group_col, sort=True).size()
    group_sizes = group_sizes.astype(int)
    total_kept = int(group_sizes.sum())
    if total_kept <= 0:
        raise ValueError('no kept samples available for long-tail summary')

    if epoch_num_samples is None:
        epoch_num_samples = total_kept
    epoch_num_samples = int(epoch_num_samples)

    rho_map = compute_group_headness(group_sizes)
    natural_probs = group_sizes.astype(np.float64) / float(total_kept)
    balanced_logits = group_sizes.astype(np.float64).pow(float(alpha))
    balanced_probs = balanced_logits / float(balanced_logits.sum())

    rows = []
    for group_id, kept_group_size in group_sizes.items():
        prune_info = None if prune_group_counts is None else prune_group_counts.get(str(group_id))
        raw_group_size = int(prune_info['group_size']) if prune_info is not None and 'group_size' in prune_info else int(kept_group_size)
        removed = int(prune_info['removed']) if prune_info is not None and 'removed' in prune_info else max(0, raw_group_size - int(kept_group_size))
        requested_remove = int(prune_info['requested_remove']) if prune_info is not None and 'requested_remove' in prune_info else removed
        rho_g = float(rho_map.get(group_id, 0.0))
        natural_prob = float(natural_probs.loc[group_id])
        balanced_prob = float(balanced_probs.loc[group_id])
        prob_ratio = float(balanced_prob / natural_prob) if natural_prob > 0 else None
        rows.append({
            group_col: group_id,
            'pseudo_group_id': group_id,
            'raw_group_size': int(raw_group_size),
            'kept_group_size': int(kept_group_size),
            'removed_group_size': int(removed),
            'requested_remove': int(requested_remove),
            'rho_g': rho_g,
            'sampling_prob_natural': natural_prob,
            'sampling_prob_balanced': balanced_prob,
            'prob_ratio': prob_ratio,
            'expected_samples_per_epoch': float(balanced_prob * epoch_num_samples),
            'is_tail_group': int(rho_g < float(tail_rho_thr)),
            'is_medium_group': int(float(tail_rho_thr) <= rho_g < float(head_rho_thr)),
            'is_head_group': int(rho_g >= float(head_rho_thr)),
            'num_tail_aug_candidates': 0,
            'num_tail_aug_selected': 0,
        })

    group_summary_df = pd.DataFrame(rows)
    if group_col != 'group_id':
        group_summary_df['group_id'] = group_summary_df[group_col]
    return group_summary_df.sort_values(group_col).reset_index(drop=True)


def _build_sample_key_series(df):
    if 'sample_key' in df.columns:
        return df['sample_key'].astype(str)
    return (
        df['class_id'].astype(int).astype(str)
        + '::'
        + df['base_idx'].astype(int).astype(str)
        + '::'
        + df['sample_idx'].astype(int).astype(str)
    )


def build_group_power_sample_weights(kept_samples_df, group_summary_df, group_col='group_id',
                                     alpha=0.5, clip_min=0.25, clip_max=4.0,
                                     weight_prefix='sampler_weight'):
    if group_col not in kept_samples_df.columns:
        raise ValueError('long-tail group column not found: {}'.format(group_col))
    ordered_df = _sort_kept_samples(kept_samples_df)
    if len(ordered_df) == 0:
        raise ValueError('kept_samples_df is empty')

    group_size_map = {
        row[group_col]: int(row['kept_group_size'])
        for _, row in group_summary_df.iterrows()
    }
    rho_map = {
        row[group_col]: float(row['rho_g'])
        for _, row in group_summary_df.iterrows()
    }
    natural_prob_map = {
        row[group_col]: float(row['sampling_prob_natural'])
        for _, row in group_summary_df.iterrows()
    }
    balanced_prob_map = {
        row[group_col]: float(row['sampling_prob_balanced'])
        for _, row in group_summary_df.iterrows()
    }

    raw_weights = np.array([
        float(group_size_map[group_id]) ** (float(alpha) - 1.0)
        for group_id in ordered_df[group_col].tolist()
    ], dtype=np.float64)
    mean_weight = float(raw_weights.mean()) if len(raw_weights) > 0 else 1.0
    if mean_weight <= 0:
        raise ValueError('invalid raw sampler weights')
    normalized_weights = raw_weights / mean_weight
    clipped_weights = np.clip(normalized_weights, float(clip_min), float(clip_max))

    output_df = ordered_df.copy()
    if group_col != 'group_id' and 'group_id' not in output_df.columns:
        output_df['group_id'] = output_df[group_col]
    if 'pseudo_group_id' not in output_df.columns:
        output_df['pseudo_group_id'] = output_df[group_col]
    output_df['selected_group_col'] = str(group_col)
    output_df['kept_group_size'] = output_df[group_col].map(group_size_map).astype(int)
    output_df['rho_g'] = output_df[group_col].map(rho_map).astype(float)
    output_df['sampling_prob_natural'] = output_df[group_col].map(natural_prob_map).astype(float)
    output_df['sampling_prob_balanced'] = output_df[group_col].map(balanced_prob_map).astype(float)
    output_df['sample_key'] = _build_sample_key_series(output_df)
    output_df['sample_key'] = output_df['sample_key'].astype(str)
    output_df['sample_idx'] = output_df['sample_idx'].astype(int)
    output_df['class_id'] = output_df['class_id'].astype(int)
    output_df['base_idx'] = output_df['base_idx'].astype(int)
    output_df['pseudo_group_id'] = output_df['pseudo_group_id'].astype(int)
    output_df['{}{}'.format(weight_prefix, '_raw')] = raw_weights.astype(float)
    output_df['{}{}'.format(weight_prefix, '_before_clip')] = normalized_weights.astype(float)
    output_df[weight_prefix] = clipped_weights.astype(float)
    output_df['{}{}'.format(weight_prefix, '_clip_min_hit')] = (normalized_weights < float(clip_min)).astype(int)
    output_df['{}{}'.format(weight_prefix, '_clip_max_hit')] = (normalized_weights > float(clip_max)).astype(int)
    output_df['is_tail_reliable'] = 0
    output_df['is_augmented'] = 0
    output_df['aug_parent_key'] = ''
    output_df['aug_type'] = ''
    output_df['aug_seed'] = -1
    output_df['aug_rank'] = -1
    return output_df.reset_index(drop=True)


def build_longtail_sampling_plan(kept_samples_df, args, original_num_samples=None, prune_group_counts=None):
    ordered_kept_samples_df = _sort_kept_samples(kept_samples_df)
    if len(ordered_kept_samples_df) == 0:
        raise ValueError('kept_samples_df is empty')

    epoch_mode = str(args.lt_epoch_num_samples)
    if epoch_mode == 'kept':
        epoch_num_samples = int(len(ordered_kept_samples_df))
    elif epoch_mode == 'original':
        if original_num_samples is None:
            raise ValueError('original_num_samples is required when lt_epoch_num_samples=original')
        epoch_num_samples = int(original_num_samples)
    else:
        raise ValueError('unsupported lt_epoch_num_samples: {}'.format(epoch_mode))

    group_summary_df = build_longtail_group_summary(
        ordered_kept_samples_df,
        group_col=args.lt_group_col,
        alpha=args.lt_sampler_alpha,
        epoch_num_samples=epoch_num_samples,
        prune_group_counts=prune_group_counts,
        tail_rho_thr=args.lt_tail_rho_thr,
    )
    sample_weights_df = build_group_power_sample_weights(
        ordered_kept_samples_df,
        group_summary_df,
        group_col=args.lt_group_col,
        alpha=args.lt_sampler_alpha,
        clip_min=args.lt_sampler_clip_min,
        clip_max=args.lt_sampler_clip_max,
        weight_prefix='sampler_weight',
    )
    loss_weights_df = build_group_power_sample_weights(
        ordered_kept_samples_df,
        group_summary_df,
        group_col=args.lt_group_col,
        alpha=args.lt_sampler_alpha,
        clip_min=args.lt_sampler_clip_min,
        clip_max=args.lt_sampler_clip_max,
        weight_prefix='loss_weight',
    )
    merge_columns = [
        'sample_key',
        'loss_weight_raw',
        'loss_weight_before_clip',
        'loss_weight',
        'loss_weight_clip_min_hit',
        'loss_weight_clip_max_hit',
    ]
    sample_weights_df = sample_weights_df.merge(loss_weights_df[merge_columns], on='sample_key', how='left')

    reliability_column = 'p_high' if 'p_high' in sample_weights_df.columns else None
    reliability_mode = 'p_high' if reliability_column is not None else 'score_fallback'
    if reliability_column is None and 'image_score' not in sample_weights_df.columns:
        raise ValueError('phase6 tail augmentation requires p_high or image_score in kept_samples_df')

    tail_group_ids = set(group_summary_df.loc[group_summary_df['is_tail_group'] == 1, args.lt_group_col].tolist())
    sample_weights_df['is_tail_reliable'] = 0
    tail_aug_rows = []
    next_aug_sample_idx = int(sample_weights_df['sample_idx'].max()) + 1 if len(sample_weights_df) > 0 else 0

    for group_id in sorted(tail_group_ids):
        group_mask = sample_weights_df[args.lt_group_col] == group_id
        group_df = sample_weights_df.loc[group_mask].copy().reset_index(drop=True)
        if len(group_df) == 0:
            continue
        max_group_aug = max(0, int(math.floor(len(group_df) * float(args.lt_tail_aug_max_group_ratio))))
        if max_group_aug == 0:
            continue

        if reliability_column is not None:
            group_df['tail_reliability_score'] = 1.0 - pd.to_numeric(group_df[reliability_column], errors='coerce').fillna(1.0)
            reliable_df = group_df.loc[pd.to_numeric(group_df[reliability_column], errors='coerce').fillna(1.0) <= float(args.lt_tail_aug_p_noisy_thr)].copy()
        else:
            rank_series = pd.to_numeric(group_df['image_score'], errors='coerce').rank(method='average', ascending=True, pct=True)
            group_df['tail_reliability_score'] = 1.0 - rank_series.fillna(0.0)
            reliable_df = group_df.loc[group_df['tail_reliability_score'] >= float(args.lt_tail_aug_score_eta)].copy()

        reliable_df = reliable_df.sort_values(['tail_reliability_score', 'image_score'], ascending=[False, True]).reset_index(drop=True)
        candidate_count = int(len(reliable_df))
        selected_count = min(int(args.lt_tail_aug_k), max_group_aug, candidate_count) if bool(args.lt_tail_aug_enable) else 0
        group_summary_df.loc[group_summary_df[args.lt_group_col] == group_id, 'num_tail_aug_candidates'] = candidate_count
        group_summary_df.loc[group_summary_df[args.lt_group_col] == group_id, 'num_tail_aug_selected'] = selected_count
        if candidate_count == 0 or selected_count == 0:
            continue

        selected_df = reliable_df.iloc[:selected_count].copy().reset_index(drop=True)
        if len(selected_df) == 0:
            continue
        sample_weights_df.loc[sample_weights_df['sample_key'].isin(selected_df['sample_key']), 'is_tail_reliable'] = 1
        score_std = float(selected_df['tail_reliability_score'].std(ddof=0)) if len(selected_df) > 1 else 0.0
        for aug_rank, (_, row) in enumerate(selected_df.iterrows()):
            aug_type = PHASE6_AUG_TYPES[aug_rank % len(PHASE6_AUG_TYPES)]
            aug_seed = int(row['sample_idx']) * 1009 + aug_rank * 97 + 17
            aug_sample_key = 'aug::{}'.format(next_aug_sample_idx)
            tail_aug_rows.append({
                'aug_sample_key': aug_sample_key,
                'parent_sample_key': str(row['sample_key']),
                'parent_img_path': row.get('img_path'),
                'group_id': int(row['group_id']),
                'pseudo_group_id': int(row.get('pseudo_group_id', row['group_id'])),
                'class_id': int(row['class_id']),
                'base_idx': int(row['base_idx']),
                'sample_idx': int(next_aug_sample_idx),
                'aug_seed': int(aug_seed),
                'aug_type': aug_type,
                'aug_rank': int(aug_rank),
                'origin_score': float(row['image_score']) if 'image_score' in row and pd.notna(row['image_score']) else None,
                'aug_score': float(row['tail_reliability_score']),
                'aug_score_std': score_std,
                'selected': 1,
                'is_tail_reliable': 1,
                'sampler_weight': float(row['sampler_weight']) * float(args.lt_aug_weight_scale),
                'loss_weight': float(row['loss_weight']) * float(args.lt_aug_weight_scale),
            })
            next_aug_sample_idx += 1

    tail_aug_manifest_df = pd.DataFrame(tail_aug_rows, columns=TAIL_AUG_COLUMNS)
    if len(tail_aug_manifest_df) > 0:
        parent_map = sample_weights_df.set_index('sample_key', drop=False)
        aug_rows = []
        for _, aug_row in tail_aug_manifest_df.iterrows():
            parent_row = parent_map.loc[str(aug_row['parent_sample_key'])].copy()
            parent_row['sample_key'] = str(aug_row['aug_sample_key'])
            parent_row['sample_idx'] = int(aug_row['sample_idx'])
            parent_row['is_augmented'] = 1
            parent_row['aug_parent_key'] = str(aug_row['parent_sample_key'])
            parent_row['aug_type'] = str(aug_row['aug_type'])
            parent_row['aug_seed'] = int(aug_row['aug_seed'])
            parent_row['aug_rank'] = int(aug_row['aug_rank'])
            parent_row['sampler_weight'] = float(aug_row['sampler_weight'])
            parent_row['loss_weight'] = float(aug_row['loss_weight'])
            aug_rows.append(parent_row)
        aug_rows_df = pd.DataFrame(aug_rows)
        sample_weights_df = pd.concat([sample_weights_df, aug_rows_df], ignore_index=True)

    sample_weights_df = sample_weights_df.sort_values(['class_id', 'base_idx', 'is_augmented', 'sample_idx']).reset_index(drop=True)
    sampler_weights = sample_weights_df['sampler_weight'].to_numpy(dtype=np.float64)
    summary = {
        'lt_enable': True,
        'lt_group_col': str(args.lt_group_col),
        'lt_sampler_enable': bool(args.lt_sampler_enable),
        'lt_sampler_alpha': float(args.lt_sampler_alpha),
        'lt_sampler_clip_min': float(args.lt_sampler_clip_min),
        'lt_sampler_clip_max': float(args.lt_sampler_clip_max),
        'lt_epoch_num_samples_mode': epoch_mode,
        'lt_epoch_num_samples_effective': int(epoch_num_samples),
        'lt_group_balanced_loss': bool(args.lt_group_balanced_loss),
        'lt_tail_aug_enable': bool(args.lt_tail_aug_enable),
        'lt_tail_rho_thr': float(args.lt_tail_rho_thr),
        'lt_tail_aug_p_noisy_thr': float(args.lt_tail_aug_p_noisy_thr),
        'lt_tail_aug_k': int(args.lt_tail_aug_k),
        'lt_tail_aug_max_group_ratio': float(args.lt_tail_aug_max_group_ratio),
        'lt_tail_aug_score_eta': float(args.lt_tail_aug_score_eta),
        'lt_aug_weight_scale': float(args.lt_aug_weight_scale),
        'lt_tail_aug_use_warmup_score': bool(args.lt_tail_aug_use_warmup_score),
        'lt_tail_aug_manifest_path': args.lt_tail_aug_manifest_path,
        'lt_tail_aug_reliability_mode': reliability_mode,
        'lt_total_kept_samples': int(len(ordered_kept_samples_df)),
        'lt_total_training_samples': int(len(sample_weights_df)),
        'lt_total_groups': int(len(group_summary_df)),
        'lt_tail_group_count': int(group_summary_df['is_tail_group'].sum()),
        'lt_head_group_count': int(group_summary_df['is_head_group'].sum()),
        'lt_sampler_weight_mean': float(sample_weights_df['sampler_weight'].mean()),
        'lt_sampler_weight_min': float(sample_weights_df['sampler_weight'].min()),
        'lt_sampler_weight_max': float(sample_weights_df['sampler_weight'].max()),
        'lt_sampler_weight_clip_min_hit_count': int(sample_weights_df['sampler_weight_clip_min_hit'].sum()),
        'lt_sampler_weight_clip_max_hit_count': int(sample_weights_df['sampler_weight_clip_max_hit'].sum()),
        'lt_loss_weight_mean': float(sample_weights_df['loss_weight'].mean()),
        'lt_loss_weight_min': float(sample_weights_df['loss_weight'].min()),
        'lt_loss_weight_max': float(sample_weights_df['loss_weight'].max()),
        'lt_loss_weight_clip_min_hit_count': int(sample_weights_df['loss_weight_clip_min_hit'].sum()),
        'lt_loss_weight_clip_max_hit_count': int(sample_weights_df['loss_weight_clip_max_hit'].sum()),
        'lt_tail_aug_candidate_count': int(group_summary_df['num_tail_aug_candidates'].sum()),
        'lt_tail_aug_selected_count': int(group_summary_df['num_tail_aug_selected'].sum()),
        'lt_group_size_min': int(group_summary_df['kept_group_size'].min()),
        'lt_group_size_max': int(group_summary_df['kept_group_size'].max()),
        'lt_group_size_mean': float(group_summary_df['kept_group_size'].mean()),
        'lt_sampler_status': 'enabled' if bool(args.lt_sampler_enable) else 'disabled',
        'lt_group_balanced_loss_status': 'enabled' if bool(args.lt_group_balanced_loss) else 'disabled',
        'lt_tail_aug_status': 'enabled' if bool(args.lt_tail_aug_enable) else 'disabled',
    }

    return {
        'group_summary_df': group_summary_df,
        'sample_weights_df': sample_weights_df,
        'tail_aug_manifest_df': tail_aug_manifest_df,
        'sampler_weights': sampler_weights,
        'epoch_num_samples': int(epoch_num_samples),
        'summary': summary,
    }


def save_longtail_artifacts(iter_dir, sampling_plan, extra_summary=None):
    os.makedirs(iter_dir, exist_ok=True)
    sampling_plan['group_summary_df'].to_csv(os.path.join(iter_dir, 'longtail_group_summary.csv'), index=False)
    sampling_plan['sample_weights_df'].to_csv(os.path.join(iter_dir, 'longtail_sample_weights.csv'), index=False)
    sampling_plan['tail_aug_manifest_df'].to_csv(os.path.join(iter_dir, 'tail_aug_manifest.csv'), index=False)

    config = dict(sampling_plan['summary'])
    if extra_summary is not None:
        config.update(_make_json_safe(extra_summary))
    with open(os.path.join(iter_dir, 'longtail_config.json'), 'w') as file:
        json.dump(_make_json_safe(config), file, indent=2)
