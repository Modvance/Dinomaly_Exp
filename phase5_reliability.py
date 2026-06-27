import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from feature_grouping import prepare_gbps_groups


DEFAULT_PHASE5_PATCH_GRID = (28, 28)


def parse_phase5_patch_grid_size(value):
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(',') if part.strip()]
        if len(parts) == 1:
            size = int(parts[0])
            return size, size
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
        raise ValueError('invalid phase5 patch grid size: {}'.format(value))
    if isinstance(value, (tuple, list)):
        if len(value) == 1:
            size = int(value[0])
            return size, size
        if len(value) == 2:
            return int(value[0]), int(value[1])
        raise ValueError('invalid phase5 patch grid size: {}'.format(value))
    size = int(value)
    return size, size


def _normalize_tensor_last_dim(tensor, eps=1e-12):
    return F.normalize(tensor.float(), p=2, dim=-1, eps=eps)


def _safe_mad(values, eps):
    values = np.asarray(values, dtype=np.float32)
    median = float(np.median(values)) if values.size > 0 else 0.0
    mad = float(np.median(np.abs(values - median))) if values.size > 0 else 0.0
    return median, mad + float(eps)


def _compute_positive_z(values, eps):
    median, mad = _safe_mad(values, eps)
    z = (np.asarray(values, dtype=np.float32) - median) / mad
    z_pos = np.maximum(z, 0.0)
    return z_pos.astype(np.float32), median, mad


def extract_phase5_features(model, dataloader, device, patch_grid_size, eps=1e-12):
    patch_h, patch_w = parse_phase5_patch_grid_size(patch_grid_size)
    was_training = model.training
    rows = []
    image_features = []
    patch_features = []

    model.eval()
    with torch.no_grad():
        for images, _, meta in dataloader:
            images = images.to(device)
            en, _ = model(images)
            feature_map = en[-1]
            pooled = F.adaptive_avg_pool2d(feature_map, output_size=1).flatten(1)
            pooled = _normalize_tensor_last_dim(pooled, eps=eps)

            resized = F.interpolate(feature_map, size=(patch_h, patch_w), mode='bilinear', align_corners=False)
            patches = resized.flatten(2).transpose(1, 2).contiguous()
            patches = _normalize_tensor_last_dim(patches, eps=eps)

            batch_size = pooled.shape[0]
            for index in range(batch_size):
                row = {}
                for key, value in meta.items():
                    item = value[index]
                    if torch.is_tensor(item):
                        if item.ndim == 0:
                            item = item.item()
                        else:
                            item = item.detach().cpu()
                    elif isinstance(item, np.generic):
                        item = item.item()
                    row[key] = item
                row['sample_key'] = int(row['sample_idx'])
                rows.append(row)
                image_features.append(pooled[index].detach().cpu())
                patch_features.append(patches[index].detach().cpu())

    if was_training:
        model.train()

    metadata = pd.DataFrame(rows)
    image_features = torch.stack(image_features, dim=0) if len(image_features) > 0 else torch.zeros((0, 0), dtype=torch.float32)
    patch_features = torch.stack(patch_features, dim=0) if len(patch_features) > 0 else torch.zeros((0, patch_h * patch_w, 0), dtype=torch.float32)
    return metadata, image_features, patch_features


def compute_group_image_knn_distance(group_image_features, knn_k):
    features = np.asarray(group_image_features, dtype=np.float32)
    num_samples = int(features.shape[0])
    if num_samples <= 1:
        return np.zeros(num_samples, dtype=np.float32)
    effective_k = max(1, min(int(knn_k), num_samples - 1))
    neighbors = NearestNeighbors(n_neighbors=effective_k + 1, metric='cosine')
    neighbors.fit(features)
    distances, _ = neighbors.kneighbors(features)
    return distances[:, 1:].mean(axis=1).astype(np.float32)


def compute_image_reliability(group_image_features, knn_k, min_group_size, eps):
    group_image_features = np.asarray(group_image_features, dtype=np.float32)
    num_samples = int(group_image_features.shape[0])
    if num_samples < int(min_group_size):
        return {
            'D_img': np.zeros(num_samples, dtype=np.float32),
            'w_img': np.ones(num_samples, dtype=np.float32),
            'z_img_pos': np.zeros(num_samples, dtype=np.float32),
            'median': 0.0,
            'mad': float(eps),
            'fallback': True,
            'fallback_reason': 'small_group',
        }
    D_img = compute_group_image_knn_distance(group_image_features, knn_k)
    z_img_pos, median, mad = _compute_positive_z(D_img, eps)
    w_img = 1.0 / (1.0 + z_img_pos)
    return {
        'D_img': D_img.astype(np.float32),
        'w_img': w_img.astype(np.float32),
        'z_img_pos': z_img_pos.astype(np.float32),
        'median': float(median),
        'mad': float(mad),
        'fallback': False,
        'fallback_reason': 'none',
    }


def _select_memory_candidates(group_patch_features, w_img, top_ratio):
    num_samples = int(group_patch_features.shape[0])
    if num_samples == 0:
        return group_patch_features.reshape(0, group_patch_features.shape[-1])
    top_count = max(1, int(np.ceil(num_samples * float(top_ratio))))
    order = np.argsort(-np.asarray(w_img, dtype=np.float32))
    selected = order[:top_count]
    return group_patch_features[selected].reshape(-1, group_patch_features.shape[-1])


def build_group_patch_memory(group_patch_features, w_img, memory_size, top_ratio, random_state, build_mode='kmeans'):
    features = np.asarray(group_patch_features, dtype=np.float32)
    if features.size == 0:
        return np.zeros((0, 0), dtype=np.float32), True, 'empty_group_patch_features'
    candidates = _select_memory_candidates(features, w_img, top_ratio)
    if candidates.shape[0] == 0:
        return np.zeros((0, features.shape[-1]), dtype=np.float32), True, 'empty_memory_candidates'
    if candidates.shape[0] <= int(memory_size):
        return candidates.astype(np.float32), False, 'none'
    if str(build_mode) == 'random':
        rng = np.random.RandomState(int(random_state))
        selected = rng.choice(candidates.shape[0], size=int(memory_size), replace=False)
        memory = candidates[selected]
    else:
        kmeans = KMeans(n_clusters=int(memory_size), random_state=int(random_state), n_init=10)
        kmeans.fit(candidates)
        memory = kmeans.cluster_centers_.astype(np.float32)
    norms = np.linalg.norm(memory, axis=1, keepdims=True)
    memory = memory / np.clip(norms, a_min=1e-12, a_max=None)
    return memory.astype(np.float32), False, 'none'


def compute_group_patch_distance(group_patch_features, memory):
    features = np.asarray(group_patch_features, dtype=np.float32)
    memory = np.asarray(memory, dtype=np.float32)
    num_samples = int(features.shape[0])
    num_patches = int(features.shape[1]) if features.ndim >= 2 else 0
    if memory.size == 0 or num_samples == 0 or num_patches == 0:
        return np.zeros((num_samples, num_patches), dtype=np.float32)
    flat = features.reshape(-1, features.shape[-1])
    sim = np.matmul(flat, memory.T)
    dist = 1.0 - np.max(sim, axis=1)
    return dist.reshape(num_samples, num_patches).astype(np.float32)


def compute_patch_reliability(group_patch_features, memory, min_group_size, eps):
    features = np.asarray(group_patch_features, dtype=np.float32)
    num_samples = int(features.shape[0])
    num_patches = int(features.shape[1]) if features.ndim >= 2 else 0
    if num_samples < int(min_group_size):
        return {
            'D_patch': np.zeros((num_samples, num_patches), dtype=np.float32),
            'w_patch': np.ones((num_samples, num_patches), dtype=np.float32),
            'z_patch_pos': np.zeros((num_samples, num_patches), dtype=np.float32),
            'median': 0.0,
            'mad': float(eps),
            'fallback': True,
            'fallback_reason': 'small_group',
        }
    if memory.size == 0:
        return {
            'D_patch': np.zeros((num_samples, num_patches), dtype=np.float32),
            'w_patch': np.ones((num_samples, num_patches), dtype=np.float32),
            'z_patch_pos': np.zeros((num_samples, num_patches), dtype=np.float32),
            'median': 0.0,
            'mad': float(eps),
            'fallback': True,
            'fallback_reason': 'empty_memory',
        }
    D_patch = compute_group_patch_distance(features, memory)
    z_patch_pos, median, mad = _compute_positive_z(D_patch.reshape(-1), eps)
    z_patch_pos = z_patch_pos.reshape(D_patch.shape)
    w_patch = 1.0 / (1.0 + z_patch_pos)
    return {
        'D_patch': D_patch.astype(np.float32),
        'w_patch': w_patch.astype(np.float32),
        'z_patch_pos': z_patch_pos.astype(np.float32),
        'median': float(median),
        'mad': float(mad),
        'fallback': False,
        'fallback_reason': 'none',
    }


def build_phase5_reliability_bank(group_assignments_df, metadata, image_features, patch_features, args):
    metadata = metadata.copy()
    metadata['sample_key'] = metadata['sample_key'].astype(int)
    assignments = group_assignments_df[['sample_key', 'group_id']].copy()
    assignments['sample_key'] = assignments['sample_key'].astype(int)
    assignments['group_id'] = assignments['group_id'].astype(int)
    merged = metadata.merge(assignments, on='sample_key', how='left')
    if merged['group_id'].isna().any():
        missing_count = int(merged['group_id'].isna().sum())
        raise ValueError('missing group_id for {} samples'.format(missing_count))
    merged['group_id'] = merged['group_id'].astype(int)

    image_features_np = image_features.detach().cpu().numpy().astype(np.float32)
    patch_features_np = patch_features.detach().cpu().numpy().astype(np.float32)
    patch_grid_size = parse_phase5_patch_grid_size(getattr(args, 'phase5_patch_grid_size', DEFAULT_PHASE5_PATCH_GRID))
    reliability_bank = {}
    rows = []

    for group_id, group_df in merged.groupby('group_id', sort=True):
        group_indices = group_df.index.to_numpy(dtype=int)
        group_image_features = image_features_np[group_indices]
        group_patch_features = patch_features_np[group_indices]
        image_result = compute_image_reliability(
            group_image_features,
            knn_k=getattr(args, 'phase5_knn_k', 5),
            min_group_size=getattr(args, 'phase5_min_group_size', 3),
            eps=getattr(args, 'phase5_eps', 1e-6),
        )
        memory, memory_fallback, memory_fallback_reason = build_group_patch_memory(
            group_patch_features,
            image_result['w_img'],
            memory_size=getattr(args, 'phase5_memory_size', 1024),
            top_ratio=getattr(args, 'phase5_top_wimg_ratio', 0.5),
            random_state=int(group_id),
            build_mode=getattr(args, 'phase5_memory_build_mode', 'kmeans'),
        )
        patch_result = compute_patch_reliability(
            group_patch_features,
            memory,
            min_group_size=getattr(args, 'phase5_min_group_size', 3),
            eps=getattr(args, 'phase5_eps', 1e-6),
        )

        if image_result['fallback_reason'] != 'none':
            fallback_reason = image_result['fallback_reason']
        elif memory_fallback_reason != 'none':
            fallback_reason = memory_fallback_reason
        elif patch_result['fallback_reason'] != 'none':
            fallback_reason = patch_result['fallback_reason']
        else:
            fallback_reason = 'none'

        for local_pos, (_, sample_row) in enumerate(group_df.reset_index().iterrows()):
            sample_key = int(sample_row['sample_key'])
            w_patch = patch_result['w_patch'][local_pos].reshape(patch_grid_size)
            D_patch = patch_result['D_patch'][local_pos].reshape(patch_grid_size)
            reliability_bank[sample_key] = {
                'sample_idx': int(sample_key),
                'group_id': int(group_id),
                'w_img': float(image_result['w_img'][local_pos]),
                'w_patch': torch.tensor(w_patch, dtype=torch.float32),
                'D_img': float(image_result['D_img'][local_pos]),
                'D_patch': torch.tensor(D_patch, dtype=torch.float32),
                'fallback': bool(image_result['fallback'] or patch_result['fallback'] or memory_fallback),
                'fallback_reason': str(fallback_reason),
            }
            row = {
                'sample_key': int(sample_key),
                'sample_idx': int(sample_key),
                'group_id': int(group_id),
                'group_size': int(len(group_df)),
                'w_img': float(image_result['w_img'][local_pos]),
                'D_img': float(image_result['D_img'][local_pos]),
                'patch_h': int(patch_grid_size[0]),
                'patch_w': int(patch_grid_size[1]),
                'patch_count': int(patch_grid_size[0] * patch_grid_size[1]),
                'mean_w_patch': float(np.mean(w_patch)),
                'min_w_patch': float(np.min(w_patch)),
                'mean_D_patch': float(np.mean(D_patch)),
                'max_D_patch': float(np.max(D_patch)),
                'suppress_ratio': float(np.mean(w_patch < 1.0)),
                'image_fallback': bool(image_result['fallback']),
                'patch_fallback': bool(patch_result['fallback'] or memory_fallback),
                'fallback_reason': str(fallback_reason),
                'img_path': sample_row.get('img_path'),
            }
            for optional_column in ['class_id', 'class_name', 'base_idx', 'is_contaminated']:
                if optional_column in sample_row.index:
                    row[optional_column] = sample_row[optional_column]
            rows.append(row)

    summary_df = pd.DataFrame(rows)
    global_summary = {
        'num_samples': int(len(summary_df)),
        'num_groups': int(merged['group_id'].nunique()),
        'phase5_knn_k': int(getattr(args, 'phase5_knn_k', 5)),
        'phase5_memory_size': int(getattr(args, 'phase5_memory_size', 1024)),
        'phase5_top_wimg_ratio': float(getattr(args, 'phase5_top_wimg_ratio', 0.5)),
        'phase5_min_group_size': int(getattr(args, 'phase5_min_group_size', 3)),
        'phase5_patch_grid_size': [int(patch_grid_size[0]), int(patch_grid_size[1])],
        'mean_w_img': float(summary_df['w_img'].mean()) if len(summary_df) > 0 else 1.0,
        'min_w_img': float(summary_df['w_img'].min()) if len(summary_df) > 0 else 1.0,
        'mean_w_patch': float(summary_df['mean_w_patch'].mean()) if len(summary_df) > 0 else 1.0,
        'min_w_patch': float(summary_df['min_w_patch'].min()) if len(summary_df) > 0 else 1.0,
        'mean_suppress_ratio': float(summary_df['suppress_ratio'].mean()) if len(summary_df) > 0 else 0.0,
        'fallback_sample_count': int((summary_df['fallback_reason'].fillna('none') != 'none').sum()) if len(summary_df) > 0 else 0,
    }
    return reliability_bank, summary_df, global_summary


def prepare_phase5_reliability(model, train_eval_dataloader, device, save_dir, args):
    phase5_group_dir = os.path.join(save_dir, 'phase5_groups')
    os.makedirs(phase5_group_dir, exist_ok=True)
    group_assignments_df, group_info = prepare_gbps_groups(model, train_eval_dataloader, device, phase5_group_dir, args)

    cache_path = getattr(args, 'phase5_reliability_cache_path', None)
    if cache_path is None:
        reliability_dir = os.path.join(save_dir, 'phase5_reliability')
        cache_path = os.path.join(reliability_dir, 'reliability_bank.pt')
    else:
        reliability_dir = os.path.dirname(os.path.abspath(cache_path))
    os.makedirs(reliability_dir, exist_ok=True)

    summary_csv_path = os.path.join(reliability_dir, 'reliability_summary.csv')
    summary_json_path = os.path.join(reliability_dir, 'reliability_summary.json')
    force_recompute = bool(getattr(args, 'phase5_force_recompute', False))

    if (not force_recompute) and os.path.isfile(cache_path) and os.path.isfile(summary_csv_path):
        reliability_bank = torch.load(cache_path, map_location='cpu')
        summary_df = pd.read_csv(summary_csv_path)
        if os.path.isfile(summary_json_path):
            import json
            with open(summary_json_path) as file:
                global_summary = json.load(file)
        else:
            global_summary = {}
        global_summary.update({
            'cache_path': os.path.abspath(cache_path),
            'loaded_from_cache': True,
            'group_info': group_info,
        })
        return {
            'group_assignments_df': group_assignments_df,
            'group_info': group_info,
            'reliability_bank': reliability_bank,
            'summary_df': summary_df,
            'global_summary': global_summary,
            'cache_path': os.path.abspath(cache_path),
            'reliability_dir': reliability_dir,
        }

    metadata, image_features, patch_features = extract_phase5_features(
        model,
        train_eval_dataloader,
        device,
        patch_grid_size=getattr(args, 'phase5_patch_grid_size', DEFAULT_PHASE5_PATCH_GRID),
        eps=getattr(args, 'phase5_eps', 1e-6),
    )
    reliability_bank, summary_df, global_summary = build_phase5_reliability_bank(
        group_assignments_df,
        metadata,
        image_features,
        patch_features,
        args,
    )
    torch.save(reliability_bank, cache_path)
    summary_df.to_csv(summary_csv_path, index=False)
    global_summary.update({
        'cache_path': os.path.abspath(cache_path),
        'loaded_from_cache': False,
        'group_info': group_info,
    })
    with open(summary_json_path, 'w') as file:
        import json
        json.dump(global_summary, file, indent=2)
    return {
        'group_assignments_df': group_assignments_df,
        'group_info': group_info,
        'reliability_bank': reliability_bank,
        'summary_df': summary_df,
        'global_summary': global_summary,
        'cache_path': os.path.abspath(cache_path),
        'reliability_dir': reliability_dir,
    }
