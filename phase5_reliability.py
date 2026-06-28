import gc
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from feature_grouping import prepare_gbps_groups


DEFAULT_PHASE5_PATCH_GRID = (28, 28)
DEFAULT_PHASE5_FAISS_BATCH_SIZE = 200000


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


def _as_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(int(value))
    value = str(value).strip().lower()
    if value in ['1', 'true', 'yes', 'y', 'on']:
        return True
    if value in ['0', 'false', 'no', 'n', 'off']:
        return False
    return bool(default)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


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


def _clip_cosine_distance(distance):
    distance = np.asarray(distance, dtype=np.float32)
    return np.clip(distance, 0.0, 2.0).astype(np.float32)


def l2_normalize_np(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x.astype(np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, float(eps))


def check_faiss_available(require_gpu=True, gpu_id=0):
    try:
        import faiss  # noqa: F401
    except ImportError:
        return False, 'faiss is not installed'

    if require_gpu:
        try:
            import faiss
            num_gpus = int(faiss.get_num_gpus())
            if num_gpus <= 0:
                return False, 'faiss is installed but no GPU is available'
            if int(gpu_id) >= num_gpus:
                return False, 'requested gpu_id {} but faiss sees only {} gpu(s)'.format(int(gpu_id), num_gpus)
        except Exception as exc:
            return False, 'failed to check faiss gpu: {}'.format(exc)

    return True, 'faiss is available'


def _build_backend_config_signature(config):
    return {
        'phase5_use_faiss': bool(config['use_faiss_requested']),
        'phase5_faiss_gpu': bool(config['faiss_gpu_requested']),
        'phase5_faiss_gpu_id': int(config['faiss_gpu_id']),
        'phase5_knn_backend': str(config['knn_backend_requested']),
        'phase5_kmeans_backend': str(config['kmeans_backend_requested']),
        'phase5_memory_build_mode': str(config['memory_build_mode']),
        'phase5_faiss_batch_size': int(config['faiss_batch_size']),
        'phase5_kmeans_niter': int(config['kmeans_niter']),
        'phase5_kmeans_nredo': int(config['kmeans_nredo']),
        'phase5_kmeans_spherical': bool(config['kmeans_spherical']),
        'phase5_l2_normalize': bool(config['l2_normalize']),
        'phase5_fallback_to_sklearn': bool(config['fallback_to_sklearn']),
    }


def resolve_phase5_backend_config(args):
    use_faiss_requested = _as_bool(getattr(args, 'phase5_use_faiss', 1), default=True)
    faiss_gpu_requested = _as_bool(getattr(args, 'phase5_faiss_gpu', 1), default=True)
    faiss_gpu_id = int(getattr(args, 'phase5_faiss_gpu_id', 0))
    knn_backend_requested = str(getattr(args, 'phase5_knn_backend', 'faiss')).lower()
    kmeans_backend_requested = str(getattr(args, 'phase5_kmeans_backend', 'faiss')).lower()
    memory_build_mode = str(getattr(args, 'phase5_memory_build_mode', 'kmeans')).lower()
    fallback_to_sklearn = _as_bool(getattr(args, 'phase5_fallback_to_sklearn', 1), default=True)
    faiss_batch_size = int(getattr(args, 'phase5_faiss_batch_size', DEFAULT_PHASE5_FAISS_BATCH_SIZE))
    kmeans_niter = int(getattr(args, 'phase5_kmeans_niter', 300))
    kmeans_nredo = int(getattr(args, 'phase5_kmeans_nredo', 10))
    kmeans_spherical = _as_bool(getattr(args, 'phase5_kmeans_spherical', 1), default=True)
    l2_normalize = _as_bool(getattr(args, 'phase5_l2_normalize', 1), default=True)

    any_faiss_requested = use_faiss_requested and (
        knn_backend_requested == 'faiss'
        or (kmeans_backend_requested == 'faiss' and memory_build_mode == 'kmeans')
    )
    faiss_available = False
    faiss_message = 'faiss is disabled by configuration'
    if any_faiss_requested:
        faiss_available, faiss_message = check_faiss_available(
            require_gpu=faiss_gpu_requested,
            gpu_id=faiss_gpu_id,
        )

    def _resolve_requested_backend(requested_backend, stage_name):
        requested_backend = str(requested_backend).lower()
        if requested_backend != 'faiss':
            return requested_backend, 'none'
        if use_faiss_requested and faiss_available:
            return 'faiss', 'none'
        if fallback_to_sklearn:
            return 'sklearn', '{}: {}'.format(stage_name, faiss_message)
        raise RuntimeError('phase5 {} requested FAISS but unavailable: {}'.format(stage_name, faiss_message))

    knn_backend_effective, knn_fallback_reason = _resolve_requested_backend(knn_backend_requested, 'knn')
    if memory_build_mode == 'random':
        kmeans_backend_effective = 'random'
        kmeans_fallback_reason = 'none'
    else:
        kmeans_backend_effective, kmeans_fallback_reason = _resolve_requested_backend(kmeans_backend_requested, 'kmeans')

    patch_backend_effective = 'faiss' if knn_backend_effective == 'faiss' else 'numpy'
    config = {
        'use_faiss_requested': bool(use_faiss_requested),
        'faiss_gpu_requested': bool(faiss_gpu_requested),
        'faiss_gpu_id': int(faiss_gpu_id),
        'faiss_available': bool(faiss_available),
        'faiss_message': str(faiss_message),
        'fallback_to_sklearn': bool(fallback_to_sklearn),
        'knn_backend_requested': str(knn_backend_requested),
        'knn_backend_effective': str(knn_backend_effective),
        'knn_backend_fallback_reason': str(knn_fallback_reason),
        'kmeans_backend_requested': str(kmeans_backend_requested),
        'kmeans_backend_effective': str(kmeans_backend_effective),
        'kmeans_backend_fallback_reason': str(kmeans_fallback_reason),
        'patch_backend_requested': 'faiss' if knn_backend_requested == 'faiss' else 'numpy',
        'patch_backend_effective': str(patch_backend_effective),
        'faiss_batch_size': int(faiss_batch_size),
        'kmeans_niter': int(kmeans_niter),
        'kmeans_nredo': int(kmeans_nredo),
        'kmeans_spherical': bool(kmeans_spherical),
        'l2_normalize': bool(l2_normalize),
        'memory_build_mode': str(memory_build_mode),
    }
    config['config_signature'] = _build_backend_config_signature(config)
    return config


def _format_backend_runtime_fallback(stage_name, exc):
    return '{} runtime fallback: {}'.format(stage_name, str(exc))[:240]


def sklearn_knn_mean_cosine_distance(features, k):
    features = np.asarray(features, dtype=np.float32)
    neighbors = NearestNeighbors(n_neighbors=int(k) + 1, metric='cosine')
    neighbors.fit(features)
    distances, _ = neighbors.kneighbors(features)
    return distances[:, 1:int(k) + 1].mean(axis=1).astype(np.float32)


def faiss_knn_mean_cosine_distance(features, k=5, use_gpu=True, gpu_id=0, batch_size=DEFAULT_PHASE5_FAISS_BATCH_SIZE, eps=1e-12):
    import faiss

    features = np.asarray(features, dtype=np.float32)
    features = l2_normalize_np(features, eps=eps)
    num_samples, dim = features.shape
    if num_samples == 0:
        return np.zeros((0,), dtype=np.float32)

    if use_gpu:
        resources = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = int(gpu_id)
        index = faiss.GpuIndexFlatIP(resources, dim, cfg)
    else:
        resources = None
        index = faiss.IndexFlatIP(dim)
    index.add(features)

    batches = []
    for start in range(0, num_samples, int(batch_size)):
        end = min(start + int(batch_size), num_samples)
        sims, _ = index.search(features[start:end], int(k) + 1)
        sims = sims[:, 1:int(k) + 1]
        dist = _clip_cosine_distance(1.0 - sims)
        batches.append(dist.mean(axis=1).astype(np.float32))

    del index
    if resources is not None:
        del resources
    return np.concatenate(batches, axis=0).astype(np.float32)


def sklearn_kmeans_memory(patch_features, memory_size, random_state, nredo=10):
    patch_features = np.asarray(patch_features, dtype=np.float32)
    kmeans = KMeans(
        n_clusters=int(memory_size),
        random_state=int(random_state),
        n_init=int(nredo),
    )
    kmeans.fit(patch_features)
    return kmeans.cluster_centers_.astype(np.float32)


def faiss_kmeans_memory(patch_features, memory_size=1024, niter=300, nredo=10, spherical=True,
                        use_gpu=True, gpu_id=0, seed=0, eps=1e-12, verbose=False):
    import faiss

    patch_features = np.asarray(patch_features, dtype=np.float32)
    if patch_features.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if bool(spherical):
        patch_features = l2_normalize_np(patch_features, eps=eps)

    _, dim = patch_features.shape
    kmeans = faiss.Kmeans(
        d=int(dim),
        k=int(memory_size),
        niter=int(niter),
        nredo=int(nredo),
        verbose=bool(verbose),
        gpu=bool(use_gpu),
        seed=int(seed),
        spherical=bool(spherical),
    )
    kmeans.train(patch_features)
    centroids = np.asarray(kmeans.centroids, dtype=np.float32)
    if bool(spherical):
        centroids = l2_normalize_np(centroids, eps=eps)
    return centroids.astype(np.float32)


def cpu_nearest_memory_cosine_distance(flat_patch_features, memory):
    flat_patch_features = np.asarray(flat_patch_features, dtype=np.float32)
    memory = np.asarray(memory, dtype=np.float32)
    if flat_patch_features.size == 0:
        return np.zeros((0,), dtype=np.float32)
    sim = np.matmul(flat_patch_features, memory.T)
    dist = 1.0 - np.max(sim, axis=1)
    return _clip_cosine_distance(dist)


def faiss_nearest_memory_cosine_distance(patch_features, memory, use_gpu=True, gpu_id=0,
                                         batch_size=DEFAULT_PHASE5_FAISS_BATCH_SIZE, eps=1e-12):
    import faiss

    patch_features = np.asarray(patch_features, dtype=np.float32)
    memory = np.asarray(memory, dtype=np.float32)
    if patch_features.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    patch_features = l2_normalize_np(patch_features, eps=eps)
    memory = l2_normalize_np(memory, eps=eps)
    _, dim = patch_features.shape

    if use_gpu:
        resources = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = int(gpu_id)
        index = faiss.GpuIndexFlatIP(resources, dim, cfg)
    else:
        resources = None
        index = faiss.IndexFlatIP(dim)
    index.add(memory)

    all_dist = []
    all_ids = []
    for start in range(0, patch_features.shape[0], int(batch_size)):
        end = min(start + int(batch_size), patch_features.shape[0])
        sims, ids = index.search(patch_features[start:end], 1)
        dist = _clip_cosine_distance(1.0 - sims[:, 0])
        all_dist.append(dist.astype(np.float32))
        all_ids.append(ids[:, 0].astype(np.int64))

    del index
    if resources is not None:
        del resources
    return np.concatenate(all_dist, axis=0), np.concatenate(all_ids, axis=0)


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


def compute_group_image_knn_distance(group_image_features, knn_k, backend_config=None, eps=1e-12):
    features = np.asarray(group_image_features, dtype=np.float32)
    num_samples = int(features.shape[0])
    if num_samples <= 1:
        return np.zeros(num_samples, dtype=np.float32), {
            'backend': backend_config['knn_backend_effective'] if backend_config is not None else 'sklearn',
            'requested_backend': backend_config['knn_backend_requested'] if backend_config is not None else 'sklearn',
            'duration_seconds': 0.0,
            'backend_fallback_reason': 'none',
        }

    effective_k = max(1, min(int(knn_k), num_samples - 1))
    backend = 'sklearn' if backend_config is None else str(backend_config['knn_backend_effective'])
    requested_backend = 'sklearn' if backend_config is None else str(backend_config['knn_backend_requested'])
    fallback_reason = 'none' if backend_config is None else str(backend_config.get('knn_backend_fallback_reason', 'none'))
    start_time = time.time()

    if backend == 'faiss':
        try:
            distances = faiss_knn_mean_cosine_distance(
                features,
                k=effective_k,
                use_gpu=bool(backend_config['faiss_gpu_requested']),
                gpu_id=int(backend_config['faiss_gpu_id']),
                batch_size=int(backend_config['faiss_batch_size']),
                eps=eps,
            )
        except Exception as exc:
            if backend_config is None or (not backend_config['fallback_to_sklearn']):
                raise
            backend = 'sklearn'
            fallback_reason = _format_backend_runtime_fallback('knn', exc)
            distances = sklearn_knn_mean_cosine_distance(features, effective_k)
    else:
        distances = sklearn_knn_mean_cosine_distance(features, effective_k)

    return distances.astype(np.float32), {
        'backend': str(backend),
        'requested_backend': str(requested_backend),
        'duration_seconds': float(time.time() - start_time),
        'backend_fallback_reason': str(fallback_reason),
    }


def compute_image_reliability(group_image_features, knn_k, min_group_size, eps, backend_config=None):
    group_image_features = np.asarray(group_image_features, dtype=np.float32)
    num_samples = int(group_image_features.shape[0])
    requested_backend = backend_config['knn_backend_requested'] if backend_config is not None else 'sklearn'
    effective_backend = backend_config['knn_backend_effective'] if backend_config is not None else 'sklearn'
    fallback_reason = backend_config.get('knn_backend_fallback_reason', 'none') if backend_config is not None else 'none'
    if num_samples < int(min_group_size):
        return {
            'D_img': np.zeros(num_samples, dtype=np.float32),
            'w_img': np.ones(num_samples, dtype=np.float32),
            'z_img_pos': np.zeros(num_samples, dtype=np.float32),
            'median': 0.0,
            'mad': float(eps),
            'fallback': True,
            'fallback_reason': 'small_group',
            'backend': str(effective_backend),
            'backend_requested': str(requested_backend),
            'backend_fallback_reason': str(fallback_reason),
            'backend_duration_seconds': 0.0,
        }
    D_img, knn_meta = compute_group_image_knn_distance(
        group_image_features,
        knn_k,
        backend_config=backend_config,
        eps=eps,
    )
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
        'backend': str(knn_meta['backend']),
        'backend_requested': str(knn_meta['requested_backend']),
        'backend_fallback_reason': str(knn_meta['backend_fallback_reason']),
        'backend_duration_seconds': float(knn_meta['duration_seconds']),
    }


def _select_memory_candidates(group_patch_features, w_img, top_ratio):
    num_samples = int(group_patch_features.shape[0])
    if num_samples == 0:
        return group_patch_features.reshape(0, group_patch_features.shape[-1])
    top_count = max(1, int(np.ceil(num_samples * float(top_ratio))))
    order = np.argsort(-np.asarray(w_img, dtype=np.float32))
    selected = order[:top_count]
    return group_patch_features[selected].reshape(-1, group_patch_features.shape[-1])


def build_group_patch_memory(group_patch_features, w_img, memory_size, top_ratio, random_state,
                             build_mode='kmeans', backend_config=None, eps=1e-12):
    features = np.asarray(group_patch_features, dtype=np.float32)
    build_mode = str(build_mode).lower()
    start_time = time.time()
    if features.size == 0:
        return np.zeros((0, 0), dtype=np.float32), True, 'empty_group_patch_features', {
            'backend': build_mode,
            'requested_backend': build_mode,
            'duration_seconds': 0.0,
            'backend_fallback_reason': 'none',
            'candidate_count': 0,
            'memory_size_requested': int(memory_size),
            'memory_size_actual': 0,
        }
    candidates = _select_memory_candidates(features, w_img, top_ratio)
    candidate_count = int(candidates.shape[0])
    if candidate_count == 0:
        return np.zeros((0, features.shape[-1]), dtype=np.float32), True, 'empty_memory_candidates', {
            'backend': build_mode,
            'requested_backend': build_mode,
            'duration_seconds': 0.0,
            'backend_fallback_reason': 'none',
            'candidate_count': 0,
            'memory_size_requested': int(memory_size),
            'memory_size_actual': 0,
        }
    if candidate_count <= int(memory_size):
        memory = candidates.astype(np.float32)
        norms = np.linalg.norm(memory, axis=1, keepdims=True)
        memory = memory / np.clip(norms, a_min=1e-12, a_max=None)
        return memory.astype(np.float32), False, 'none', {
            'backend': 'direct',
            'requested_backend': build_mode,
            'duration_seconds': float(time.time() - start_time),
            'backend_fallback_reason': 'none',
            'candidate_count': int(candidate_count),
            'memory_size_requested': int(memory_size),
            'memory_size_actual': int(memory.shape[0]),
        }

    backend = 'random' if build_mode == 'random' else ('sklearn' if backend_config is None else str(backend_config['kmeans_backend_effective']))
    requested_backend = 'random' if build_mode == 'random' else ('sklearn' if backend_config is None else str(backend_config['kmeans_backend_requested']))
    fallback_reason = 'none' if backend_config is None else str(backend_config.get('kmeans_backend_fallback_reason', 'none'))

    if build_mode == 'random':
        rng = np.random.RandomState(int(random_state))
        selected = rng.choice(candidate_count, size=int(memory_size), replace=False)
        memory = candidates[selected]
    elif backend == 'faiss':
        try:
            memory = faiss_kmeans_memory(
                candidates,
                memory_size=int(memory_size),
                niter=int(backend_config['kmeans_niter']),
                nredo=int(backend_config['kmeans_nredo']),
                spherical=bool(backend_config['kmeans_spherical']),
                use_gpu=bool(backend_config['faiss_gpu_requested']),
                gpu_id=int(backend_config['faiss_gpu_id']),
                seed=int(random_state),
                eps=eps,
                verbose=False,
            )
        except Exception as exc:
            if backend_config is None or (not backend_config['fallback_to_sklearn']):
                raise
            backend = 'sklearn'
            fallback_reason = _format_backend_runtime_fallback('kmeans', exc)
            memory = sklearn_kmeans_memory(
                candidates,
                memory_size=int(memory_size),
                random_state=int(random_state),
                nredo=int(backend_config['kmeans_nredo']),
            )
    else:
        nredo = 10 if backend_config is None else int(backend_config['kmeans_nredo'])
        memory = sklearn_kmeans_memory(
            candidates,
            memory_size=int(memory_size),
            random_state=int(random_state),
            nredo=int(nredo),
        )

    norms = np.linalg.norm(memory, axis=1, keepdims=True)
    memory = memory / np.clip(norms, a_min=1e-12, a_max=None)
    return memory.astype(np.float32), False, 'none', {
        'backend': str(backend),
        'requested_backend': str(requested_backend),
        'duration_seconds': float(time.time() - start_time),
        'backend_fallback_reason': str(fallback_reason),
        'candidate_count': int(candidate_count),
        'memory_size_requested': int(memory_size),
        'memory_size_actual': int(memory.shape[0]),
    }


def compute_group_patch_distance(group_patch_features, memory, backend_config=None, eps=1e-12):
    features = np.asarray(group_patch_features, dtype=np.float32)
    memory = np.asarray(memory, dtype=np.float32)
    num_samples = int(features.shape[0])
    num_patches = int(features.shape[1]) if features.ndim >= 2 else 0
    requested_backend = 'numpy' if backend_config is None else str(backend_config['patch_backend_requested'])
    backend = 'numpy' if backend_config is None else str(backend_config['patch_backend_effective'])
    start_time = time.time()

    if memory.size == 0 or num_samples == 0 or num_patches == 0:
        return np.zeros((num_samples, num_patches), dtype=np.float32), {
            'backend': str(backend),
            'requested_backend': str(requested_backend),
            'duration_seconds': 0.0,
            'backend_fallback_reason': 'none',
        }

    flat = features.reshape(-1, features.shape[-1])
    fallback_reason = 'none'
    if backend == 'faiss':
        try:
            dist, _ = faiss_nearest_memory_cosine_distance(
                flat,
                memory,
                use_gpu=bool(backend_config['faiss_gpu_requested']),
                gpu_id=int(backend_config['faiss_gpu_id']),
                batch_size=int(backend_config['faiss_batch_size']),
                eps=eps,
            )
        except Exception as exc:
            if backend_config is None or (not backend_config['fallback_to_sklearn']):
                raise
            backend = 'numpy'
            fallback_reason = _format_backend_runtime_fallback('patch_search', exc)
            dist = cpu_nearest_memory_cosine_distance(flat, memory)
    else:
        dist = cpu_nearest_memory_cosine_distance(flat, memory)

    return dist.reshape(num_samples, num_patches).astype(np.float32), {
        'backend': str(backend),
        'requested_backend': str(requested_backend),
        'duration_seconds': float(time.time() - start_time),
        'backend_fallback_reason': str(fallback_reason),
    }


def compute_patch_reliability(group_patch_features, memory, min_group_size, eps, backend_config=None):
    features = np.asarray(group_patch_features, dtype=np.float32)
    num_samples = int(features.shape[0])
    num_patches = int(features.shape[1]) if features.ndim >= 2 else 0
    requested_backend = 'numpy' if backend_config is None else str(backend_config['patch_backend_requested'])
    effective_backend = 'numpy' if backend_config is None else str(backend_config['patch_backend_effective'])
    if num_samples < int(min_group_size):
        return {
            'D_patch': np.zeros((num_samples, num_patches), dtype=np.float32),
            'w_patch': np.ones((num_samples, num_patches), dtype=np.float32),
            'z_patch_pos': np.zeros((num_samples, num_patches), dtype=np.float32),
            'median': 0.0,
            'mad': float(eps),
            'fallback': True,
            'fallback_reason': 'small_group',
            'backend': str(effective_backend),
            'backend_requested': str(requested_backend),
            'backend_fallback_reason': 'none',
            'backend_duration_seconds': 0.0,
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
            'backend': str(effective_backend),
            'backend_requested': str(requested_backend),
            'backend_fallback_reason': 'none',
            'backend_duration_seconds': 0.0,
        }
    D_patch, patch_meta = compute_group_patch_distance(
        features,
        memory,
        backend_config=backend_config,
        eps=eps,
    )
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
        'backend': str(patch_meta['backend']),
        'backend_requested': str(patch_meta['requested_backend']),
        'backend_fallback_reason': str(patch_meta['backend_fallback_reason']),
        'backend_duration_seconds': float(patch_meta['duration_seconds']),
    }


def _save_phase5_group_cache(group_cache_root, group_id, sample_keys, image_result, patch_result, memory, cache_meta):
    if group_cache_root is None:
        return
    group_dir = os.path.join(group_cache_root, 'group_{:03d}'.format(int(group_id)))
    os.makedirs(group_dir, exist_ok=True)
    np.save(os.path.join(group_dir, 'image_weight.npy'), np.asarray(image_result['w_img'], dtype=np.float32))
    np.save(os.path.join(group_dir, 'patch_weight.npy'), np.asarray(patch_result['w_patch'], dtype=np.float32))
    np.save(os.path.join(group_dir, 'image_distance.npy'), np.asarray(image_result['D_img'], dtype=np.float32))
    np.save(os.path.join(group_dir, 'patch_distance.npy'), np.asarray(patch_result['D_patch'], dtype=np.float32))
    np.save(os.path.join(group_dir, 'patch_memory.npy'), np.asarray(memory, dtype=np.float32))
    payload = dict(cache_meta)
    payload['sample_keys'] = [int(sample_key) for sample_key in sample_keys]
    with open(os.path.join(group_dir, 'meta.json'), 'w') as file:
        json.dump(_json_safe(payload), file, indent=2)


def _is_backend_cache_compatible(summary_json_data, backend_config):
    if not isinstance(summary_json_data, dict):
        return False
    cached_signature = summary_json_data.get('phase5_backend_signature')
    if cached_signature is None:
        return False
    return cached_signature == backend_config['config_signature']


def build_phase5_reliability_bank(group_assignments_df, metadata, image_features, patch_features, args,
                                  reliability_dir=None, print_fn=None):
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
    backend_config = resolve_phase5_backend_config(args)
    group_cache_root = None if reliability_dir is None else os.path.join(reliability_dir, 'phase5_cache')
    if group_cache_root is not None:
        os.makedirs(group_cache_root, exist_ok=True)

    total_image_knn_seconds = 0.0
    total_kmeans_seconds = 0.0
    total_patch_search_seconds = 0.0
    backend_fallback_count = 0

    for group_id, group_df in merged.groupby('group_id', sort=True):
        group_indices = group_df.index.to_numpy(dtype=int)
        group_image_features = image_features_np[group_indices]
        group_patch_features = patch_features_np[group_indices]
        group_start_time = time.time()
        image_result = compute_image_reliability(
            group_image_features,
            knn_k=getattr(args, 'phase5_knn_k', 5),
            min_group_size=getattr(args, 'phase5_min_group_size', 3),
            eps=getattr(args, 'phase5_eps', 1e-6),
            backend_config=backend_config,
        )
        memory, memory_fallback, memory_fallback_reason, memory_meta = build_group_patch_memory(
            group_patch_features,
            image_result['w_img'],
            memory_size=getattr(args, 'phase5_memory_size', 1024),
            top_ratio=getattr(args, 'phase5_top_wimg_ratio', 0.5),
            random_state=int(group_id),
            build_mode=getattr(args, 'phase5_memory_build_mode', 'kmeans'),
            backend_config=backend_config,
            eps=getattr(args, 'phase5_eps', 1e-6),
        )
        patch_result = compute_patch_reliability(
            group_patch_features,
            memory,
            min_group_size=getattr(args, 'phase5_min_group_size', 3),
            eps=getattr(args, 'phase5_eps', 1e-6),
            backend_config=backend_config,
        )

        total_image_knn_seconds += float(image_result['backend_duration_seconds'])
        total_kmeans_seconds += float(memory_meta['duration_seconds'])
        total_patch_search_seconds += float(patch_result['backend_duration_seconds'])

        backend_fallback_reason = 'none'
        for reason in [
            image_result.get('backend_fallback_reason', 'none'),
            memory_meta.get('backend_fallback_reason', 'none'),
            patch_result.get('backend_fallback_reason', 'none'),
        ]:
            if reason != 'none':
                backend_fallback_reason = str(reason)
                backend_fallback_count += 1
                break

        if image_result['fallback_reason'] != 'none':
            fallback_reason = image_result['fallback_reason']
        elif memory_fallback_reason != 'none':
            fallback_reason = memory_fallback_reason
        elif patch_result['fallback_reason'] != 'none':
            fallback_reason = patch_result['fallback_reason']
        else:
            fallback_reason = 'none'

        group_cache_meta = {
            'group_id': int(group_id),
            'group_size': int(len(group_df)),
            'image_backend': str(image_result['backend']),
            'memory_backend': str(memory_meta['backend']),
            'patch_backend': str(patch_result['backend']),
            'backend_fallback_reason': str(backend_fallback_reason),
            'fallback_reason': str(fallback_reason),
            'memory_candidate_count': int(memory_meta['candidate_count']),
            'memory_size_requested': int(memory_meta['memory_size_requested']),
            'memory_size_actual': int(memory_meta['memory_size_actual']),
            'image_knn_seconds': float(image_result['backend_duration_seconds']),
            'kmeans_seconds': float(memory_meta['duration_seconds']),
            'patch_search_seconds': float(patch_result['backend_duration_seconds']),
            'phase5_backend_signature': backend_config['config_signature'],
        }
        _save_phase5_group_cache(
            group_cache_root,
            group_id,
            group_df['sample_key'].tolist(),
            image_result,
            patch_result,
            memory,
            group_cache_meta,
        )

        if print_fn is not None:
            print_fn(
                'phase5 group {} size={} patches={} image_backend={} memory_backend={} patch_backend={} '
                'image_knn_s={:.2f} kmeans_s={:.2f} patch_search_s={:.2f} '
                'D_img[min/mean/max]={:.4f}/{:.4f}/{:.4f} '
                'D_patch[min/mean/max]={:.4f}/{:.4f}/{:.4f} '
                'w_img[min/mean/max]={:.4f}/{:.4f}/{:.4f} '
                'w_patch[min/mean/max]={:.4f}/{:.4f}/{:.4f}'.format(
                    int(group_id),
                    int(len(group_df)),
                    int(patch_grid_size[0] * patch_grid_size[1]),
                    image_result['backend'],
                    memory_meta['backend'],
                    patch_result['backend'],
                    float(image_result['backend_duration_seconds']),
                    float(memory_meta['duration_seconds']),
                    float(patch_result['backend_duration_seconds']),
                    float(np.min(image_result['D_img'])) if image_result['D_img'].size > 0 else 0.0,
                    float(np.mean(image_result['D_img'])) if image_result['D_img'].size > 0 else 0.0,
                    float(np.max(image_result['D_img'])) if image_result['D_img'].size > 0 else 0.0,
                    float(np.min(patch_result['D_patch'])) if patch_result['D_patch'].size > 0 else 0.0,
                    float(np.mean(patch_result['D_patch'])) if patch_result['D_patch'].size > 0 else 0.0,
                    float(np.max(patch_result['D_patch'])) if patch_result['D_patch'].size > 0 else 0.0,
                    float(np.min(image_result['w_img'])) if image_result['w_img'].size > 0 else 1.0,
                    float(np.mean(image_result['w_img'])) if image_result['w_img'].size > 0 else 1.0,
                    float(np.max(image_result['w_img'])) if image_result['w_img'].size > 0 else 1.0,
                    float(np.min(patch_result['w_patch'])) if patch_result['w_patch'].size > 0 else 1.0,
                    float(np.mean(patch_result['w_patch'])) if patch_result['w_patch'].size > 0 else 1.0,
                    float(np.max(patch_result['w_patch'])) if patch_result['w_patch'].size > 0 else 1.0,
                )
            )
            if fallback_reason != 'none' or backend_fallback_reason != 'none':
                print_fn(
                    'phase5 group {} fallback_reason={} backend_fallback_reason={}'.format(
                        int(group_id),
                        fallback_reason,
                        backend_fallback_reason,
                    )
                )

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
                'backend_fallback_reason': str(backend_fallback_reason),
                'image_backend': str(image_result['backend']),
                'memory_backend': str(memory_meta['backend']),
                'patch_backend': str(patch_result['backend']),
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
                'backend_fallback_reason': str(backend_fallback_reason),
                'image_backend': str(image_result['backend']),
                'memory_backend': str(memory_meta['backend']),
                'patch_backend': str(patch_result['backend']),
                'image_knn_seconds': float(image_result['backend_duration_seconds']),
                'kmeans_seconds': float(memory_meta['duration_seconds']),
                'patch_search_seconds': float(patch_result['backend_duration_seconds']),
                'memory_candidate_count': int(memory_meta['candidate_count']),
                'memory_size_requested': int(memory_meta['memory_size_requested']),
                'memory_size_actual': int(memory_meta['memory_size_actual']),
                'group_total_seconds': float(time.time() - group_start_time),
                'img_path': sample_row.get('img_path'),
            }
            for optional_column in ['class_id', 'class_name', 'base_idx', 'is_contaminated']:
                if optional_column in sample_row.index:
                    row[optional_column] = sample_row[optional_column]
            rows.append(row)

        del group_image_features
        del group_patch_features
        del memory
        gc.collect()
        if backend_config['faiss_gpu_requested']:
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(rows)
    global_summary = {
        'num_samples': int(len(summary_df)),
        'num_groups': int(merged['group_id'].nunique()),
        'phase5_knn_k': int(getattr(args, 'phase5_knn_k', 5)),
        'phase5_memory_size': int(getattr(args, 'phase5_memory_size', 1024)),
        'phase5_top_wimg_ratio': float(getattr(args, 'phase5_top_wimg_ratio', 0.5)),
        'phase5_min_group_size': int(getattr(args, 'phase5_min_group_size', 3)),
        'phase5_patch_grid_size': [int(patch_grid_size[0]), int(patch_grid_size[1])],
        'phase5_use_faiss': bool(backend_config['use_faiss_requested']),
        'phase5_faiss_gpu': bool(backend_config['faiss_gpu_requested']),
        'phase5_faiss_gpu_id': int(backend_config['faiss_gpu_id']),
        'phase5_faiss_batch_size': int(backend_config['faiss_batch_size']),
        'phase5_knn_backend_requested': str(backend_config['knn_backend_requested']),
        'phase5_knn_backend_effective': str(backend_config['knn_backend_effective']),
        'phase5_kmeans_backend_requested': str(backend_config['kmeans_backend_requested']),
        'phase5_kmeans_backend_effective': str(backend_config['kmeans_backend_effective']),
        'phase5_patch_backend_requested': str(backend_config['patch_backend_requested']),
        'phase5_patch_backend_effective': str(backend_config['patch_backend_effective']),
        'phase5_backend_signature': backend_config['config_signature'],
        'phase5_backend_fallback_count': int(backend_fallback_count),
        'phase5_total_image_knn_seconds': float(total_image_knn_seconds),
        'phase5_total_kmeans_seconds': float(total_kmeans_seconds),
        'phase5_total_patch_search_seconds': float(total_patch_search_seconds),
        'phase5_faiss_available': bool(backend_config['faiss_available']),
        'phase5_faiss_message': str(backend_config['faiss_message']),
        'mean_w_img': float(summary_df['w_img'].mean()) if len(summary_df) > 0 else 1.0,
        'min_w_img': float(summary_df['w_img'].min()) if len(summary_df) > 0 else 1.0,
        'mean_w_patch': float(summary_df['mean_w_patch'].mean()) if len(summary_df) > 0 else 1.0,
        'min_w_patch': float(summary_df['min_w_patch'].min()) if len(summary_df) > 0 else 1.0,
        'mean_suppress_ratio': float(summary_df['suppress_ratio'].mean()) if len(summary_df) > 0 else 0.0,
        'fallback_sample_count': int((summary_df['fallback_reason'].fillna('none') != 'none').sum()) if len(summary_df) > 0 else 0,
    }
    return reliability_bank, summary_df, global_summary


def prepare_phase5_reliability(model, train_eval_dataloader, device, save_dir, args, print_fn=None):
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
    backend_config = resolve_phase5_backend_config(args)

    if (not force_recompute) and os.path.isfile(cache_path) and os.path.isfile(summary_csv_path):
        if os.path.isfile(summary_json_path):
            with open(summary_json_path) as file:
                global_summary = json.load(file)
        else:
            global_summary = None
        if _is_backend_cache_compatible(global_summary, backend_config):
            reliability_bank = torch.load(cache_path, map_location='cpu')
            summary_df = pd.read_csv(summary_csv_path)
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
        if print_fn is not None:
            print_fn('phase5 reliability cache exists but backend signature mismatched; recomputing reliability bank')

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
        reliability_dir=reliability_dir,
        print_fn=print_fn,
    )
    torch.save(reliability_bank, cache_path)
    summary_df.to_csv(summary_csv_path, index=False)
    global_summary.update({
        'cache_path': os.path.abspath(cache_path),
        'loaded_from_cache': False,
        'group_info': group_info,
    })
    with open(summary_json_path, 'w') as file:
        json.dump(_json_safe(global_summary), file, indent=2)
    return {
        'group_assignments_df': group_assignments_df,
        'group_info': group_info,
        'reliability_bank': reliability_bank,
        'summary_df': summary_df,
        'global_summary': global_summary,
        'cache_path': os.path.abspath(cache_path),
        'reliability_dir': reliability_dir,
    }
