import json
import os
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


DEFAULT_K_CANDIDATES = (8, 10, 12, 15, 20)


def _parse_k_candidates(values):
    if values is None:
        return DEFAULT_K_CANDIDATES
    if isinstance(values, str):
        return tuple(int(part.strip()) for part in values.split(',') if part.strip())
    return tuple(int(value) for value in values)


def _to_python_scalar(value):
    if torch.is_tensor(value):
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return embeddings / norms


def _extract_encoder_gap_embeddings(model, images: torch.Tensor) -> np.ndarray:
    en, _ = model(images)
    feature_map = en[-1]
    pooled = F.adaptive_avg_pool2d(feature_map, output_size=1).flatten(1)
    return pooled.detach().cpu().numpy().astype(np.float32)


def _extract_encoder_cls_embeddings(model, images: torch.Tensor) -> np.ndarray:
    encoder = getattr(model, 'encoder', None)
    if encoder is None:
        raise ValueError('model does not expose encoder for cls embedding extraction')
    if not hasattr(encoder, 'prepare_tokens_with_masks'):
        raise ValueError('encoder_cls embedding_source requires a DINOv2-style encoder')

    x = encoder.prepare_tokens_with_masks(images)
    for blk in encoder.blocks:
        x = blk(x)
    x = encoder.norm(x)
    cls_token = x[:, 0]
    return cls_token.detach().cpu().numpy().astype(np.float32)


def extract_image_embeddings(model, dataloader, device, embedding_source='encoder'):
    if embedding_source not in {'encoder', 'encoder_cls'}:
        raise ValueError(f"unsupported embedding_source: {embedding_source}")

    was_training = model.training
    rows = []
    embedding_list = []

    model.eval()
    with torch.no_grad():
        for images, _, meta in dataloader:
            images = images.to(device)
            if embedding_source == 'encoder':
                batch_embeddings = _extract_encoder_gap_embeddings(model, images)
            else:
                batch_embeddings = _extract_encoder_cls_embeddings(model, images)

            batch_size = batch_embeddings.shape[0]
            for index in range(batch_size):
                row = {}
                for key, value in meta.items():
                    row[key] = _to_python_scalar(value[index])
                row['sample_key'] = int(row['sample_idx'])
                rows.append(row)
                embedding_list.append(batch_embeddings[index])

    if was_training:
        model.train()

    metadata = pd.DataFrame(rows)
    embeddings = np.stack(embedding_list, axis=0) if len(embedding_list) > 0 else np.zeros((0, 0), dtype=np.float32)
    return embeddings, metadata


def fit_latent_groups(
    embeddings,
    sample_keys,
    method='kmeans',
    num_groups=None,
    auto_k=True,
    k_candidates=DEFAULT_K_CANDIDATES,
    pca_dim=64,
    random_state=0,
):
    if method != 'kmeans':
        raise ValueError(f"unsupported grouping method: {method}")

    sample_keys = [int(key) for key in sample_keys]
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError('embeddings must be a 2D array')
    if embeddings.shape[0] != len(sample_keys):
        raise ValueError('embeddings and sample_keys length mismatch')
    if embeddings.shape[0] == 0:
        raise ValueError('cannot fit latent groups on empty embeddings')

    normalized = _normalize_embeddings(embeddings)
    selected_pca_dim = None
    if pca_dim is not None and pca_dim > 0 and normalized.shape[1] > pca_dim and normalized.shape[0] > pca_dim:
        selected_pca_dim = int(pca_dim)
        normalized = PCA(n_components=selected_pca_dim, random_state=random_state).fit_transform(normalized)

    if num_groups is not None:
        selected_k = int(num_groups)
        best_score = None
    else:
        candidates = _parse_k_candidates(k_candidates)
        valid_candidates = [k for k in candidates if 2 <= int(k) < len(sample_keys)]
        if len(valid_candidates) == 0:
            selected_k = max(1, min(len(sample_keys), 2))
            best_score = None
        elif not auto_k:
            selected_k = int(valid_candidates[0])
            best_score = None
        else:
            best_score = None
            selected_k = int(valid_candidates[0])
            for candidate in valid_candidates:
                kmeans = KMeans(n_clusters=int(candidate), random_state=random_state, n_init=10)
                labels = kmeans.fit_predict(normalized)
                if len(np.unique(labels)) < 2:
                    continue
                score = silhouette_score(normalized, labels)
                if best_score is None or score > best_score:
                    best_score = float(score)
                    selected_k = int(candidate)

    if selected_k <= 1:
        assignments = np.zeros(len(sample_keys), dtype=int)
    else:
        kmeans = KMeans(n_clusters=selected_k, random_state=random_state, n_init=10)
        assignments = kmeans.fit_predict(normalized)

    group_assignments = {
        int(sample_key): int(group_id)
        for sample_key, group_id in zip(sample_keys, assignments)
    }
    group_sizes = pd.Series(assignments).value_counts().sort_index().to_dict()
    group_info = {
        'method': method,
        'selected_k': int(selected_k),
        'auto_k': bool(auto_k and num_groups is None),
        'silhouette_score': None if best_score is None else float(best_score),
        'pca_dim': None if selected_pca_dim is None else int(selected_pca_dim),
        'group_sizes': {str(int(group_id)): int(size) for group_id, size in group_sizes.items()},
        'num_samples': int(len(sample_keys)),
    }
    return group_assignments, group_info


def save_group_assignments(group_assignments, metadata, save_path):
    metadata = metadata.copy()
    metadata['sample_key'] = metadata['sample_idx'].astype(int)
    metadata['group_id'] = metadata['sample_key'].map({int(key): int(value) for key, value in group_assignments.items()})
    group_sizes = metadata.groupby('group_id').size().rename('group_size').reset_index()
    metadata = metadata.merge(group_sizes, on='group_id', how='left')

    columns = ['sample_key', 'sample_idx', 'img_path', 'group_id', 'group_size']
    optional_columns = ['base_idx', 'class_name', 'class_id', 'is_contaminated']
    for column in optional_columns:
        if column in metadata.columns:
            columns.append(column)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    metadata[columns].to_csv(save_path, index=False)
    return metadata[columns].copy()


def prepare_gbps_groups(model, dataloader, device, save_dir, args):
    group_dir = os.path.join(save_dir, 'gbps_groups')
    os.makedirs(group_dir, exist_ok=True)

    if getattr(args, 'gbps_group_cache_path', None):
        cache_path = args.gbps_group_cache_path
    else:
        cache_path = os.path.join(group_dir, 'group_assignments.csv')

    summary_path = os.path.join(group_dir, 'group_summary.json')

    if os.path.isfile(cache_path):
        group_assignments_df = pd.read_csv(cache_path)
        group_assignments_df['sample_key'] = group_assignments_df['sample_key'].astype(int)
        group_assignments_df['group_id'] = group_assignments_df['group_id'].astype(int)
        group_info = {
            'cache_path': cache_path,
            'loaded_from_cache': True,
            'selected_k': int(group_assignments_df['group_id'].nunique()) if len(group_assignments_df) > 0 else 0,
            'group_sizes': {
                str(int(group_id)): int(size)
                for group_id, size in group_assignments_df.groupby('group_id').size().items()
            },
            'num_samples': int(len(group_assignments_df)),
        }
        if os.path.isfile(summary_path):
            with open(summary_path) as file:
                summary_data = json.load(file)
            group_info.update(summary_data)
        return group_assignments_df, group_info

    embeddings, metadata = extract_image_embeddings(
        model,
        dataloader,
        device,
        embedding_source=getattr(args, 'gbps_embedding_source', 'encoder'),
    )
    group_assignments, group_info = fit_latent_groups(
        embeddings,
        metadata['sample_key'].tolist(),
        method=getattr(args, 'gbps_grouping_method', 'kmeans'),
        num_groups=getattr(args, 'gbps_num_groups', None),
        auto_k=getattr(args, 'gbps_auto_k', False),
        k_candidates=getattr(args, 'gbps_k_candidates', DEFAULT_K_CANDIDATES),
        pca_dim=getattr(args, 'gbps_pca_dim', 64),
        random_state=0,
    )
    group_assignments_df = save_group_assignments(group_assignments, metadata, cache_path)
    group_info.update({
        'cache_path': cache_path,
        'loaded_from_cache': False,
        'embedding_source': getattr(args, 'gbps_embedding_source', 'encoder'),
    })
    with open(summary_path, 'w') as file:
        json.dump(group_info, file, indent=2)
    return group_assignments_df, group_info
