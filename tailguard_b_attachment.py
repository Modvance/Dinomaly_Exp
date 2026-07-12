from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


GEOMETRY_COLUMNS = [
    'group_id',
    'group_size',
    'valid_group',
    'distance_median',
    'distance_mad',
    'distance_min',
    'distance_max',
]


def _normalize_rows(tensor: torch.Tensor) -> torch.Tensor:
    return F.normalize(tensor.float(), dim=-1)


def _embedding_lookup(embeddings_payload: Dict):
    sample_indices = [int(value) for value in embeddings_payload['sample_idx']]
    embeddings = _normalize_rows(embeddings_payload['embeddings'].float())
    return {sample_idx: embeddings[index] for index, sample_idx in enumerate(sample_indices)}


def _linear_sse(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) <= 1:
        return 0.0
    design = np.stack([x, np.ones_like(x)], axis=1)
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    return float(np.sum(residual ** 2))


def _bic(sse: float, n: int, q: int) -> float:
    n = max(1, int(n))
    return float(n * np.log(float(sse) / n + 1e-12) + int(q) * np.log(n))


def build_clean_head_geometry(h_clean_df: pd.DataFrame, grouping_embeddings_payload: Dict, args) -> Tuple[Dict, pd.DataFrame]:
    clean_df = h_clean_df.copy()
    if len(clean_df) == 0:
        return {'groups': {}, 'embedding_source': grouping_embeddings_payload.get('embedding_source')}, pd.DataFrame(columns=GEOMETRY_COLUMNS)
    clean_df['sample_idx'] = clean_df['sample_idx'].astype(int)
    clean_df['group_id'] = clean_df['group_id'].astype(int)
    embedding_map = _embedding_lookup(grouping_embeddings_payload)
    min_group_size = int(getattr(args, 'tgb_min_clean_group_size', 3))

    groups = {}
    rows = []
    for group_id, group_df in clean_df.groupby('group_id', sort=True):
        vectors = []
        sample_indices = []
        for sample_idx in group_df['sample_idx'].astype(int).tolist():
            vector = embedding_map.get(int(sample_idx))
            if vector is None:
                continue
            vectors.append(vector)
            sample_indices.append(int(sample_idx))
        if len(vectors) == 0:
            continue
        matrix = torch.stack(vectors, dim=0)
        centroid = _normalize_rows(matrix.mean(dim=0, keepdim=True))[0]
        distances = (1.0 - torch.clamp(matrix @ centroid, min=-1.0, max=1.0)).cpu().numpy().astype(float)
        valid_group = len(sample_indices) >= min_group_size
        group_id = int(group_id)
        groups[group_id] = {
            'group_id': group_id,
            'centroid': centroid.cpu(),
            'distances': distances,
            'sample_indices': sample_indices,
            'valid_group': bool(valid_group),
            'group_size': int(len(sample_indices)),
        }
        rows.append({
            'group_id': group_id,
            'group_size': int(len(sample_indices)),
            'valid_group': bool(valid_group),
            'distance_median': float(np.median(distances)),
            'distance_mad': float(np.median(np.abs(distances - np.median(distances)))),
            'distance_min': float(np.min(distances)),
            'distance_max': float(np.max(distances)),
        })

    geometry = {
        'groups': groups,
        'embedding_source': grouping_embeddings_payload.get('embedding_source'),
        'min_clean_group_size': int(min_group_size),
        'num_groups': int(len(groups)),
        'num_valid_groups': int(sum(1 for entry in groups.values() if entry['valid_group'])),
    }
    return geometry, pd.DataFrame(rows, columns=GEOMETRY_COLUMNS)


def compute_tail_attachment(tail_df: pd.DataFrame, geometry: Dict, grouping_embeddings_payload: Dict):
    tail_df = tail_df.copy().reset_index(drop=True)
    if len(tail_df) == 0:
        empty_scores = tail_df.copy()
        for column in ['best_group_id', 'attachment_A', 'attachment_delta', 'attachment_score', 'best_distance', 'second_group_id']:
            empty_scores[column] = []
        return pd.DataFrame(), empty_scores

    embedding_map = _embedding_lookup(grouping_embeddings_payload)
    valid_groups = {
        int(group_id): entry
        for group_id, entry in geometry.get('groups', {}).items()
        if bool(entry.get('valid_group', False))
    }
    conformity_rows = []
    score_rows = []

    for _, row in tail_df.iterrows():
        sample_idx = int(row['sample_idx'])
        vector = embedding_map.get(sample_idx)
        p_values = []
        if vector is not None:
            for group_id, entry in sorted(valid_groups.items()):
                centroid = entry['centroid'].to(vector.device)
                distance = float(1.0 - torch.clamp(vector @ centroid, min=-1.0, max=1.0).item())
                distances = np.asarray(entry['distances'], dtype=float)
                conformity = float((1 + np.sum(distances >= distance)) / (len(distances) + 1))
                p_values.append((int(group_id), conformity, distance))
                conformity_rows.append({
                    'sample_idx': sample_idx,
                    'sample_key': int(row.get('sample_key', sample_idx)),
                    'group_id': int(group_id),
                    'distance': distance,
                    'conformity': conformity,
                })

        p_values = sorted(p_values, key=lambda item: item[1], reverse=True)
        if len(p_values) == 0:
            best_group_id = -1
            second_group_id = -1
            best_p = 0.0
            second_p = 0.0
            best_distance = float('inf')
        else:
            best_group_id, best_p, best_distance = p_values[0]
            if len(p_values) > 1:
                second_group_id, second_p, _ = p_values[1]
            else:
                second_group_id, second_p = -1, 0.0
        delta = float(best_p - second_p)
        attachment_score = float(best_p * delta)
        score_row = row.to_dict()
        score_row.update({
            'best_group_id': int(best_group_id),
            'second_group_id': int(second_group_id),
            'attachment_A': float(best_p),
            'attachment_delta': float(delta),
            'attachment_score': float(attachment_score),
            'best_distance': float(best_distance),
        })
        score_rows.append(score_row)

    return pd.DataFrame(conformity_rows), pd.DataFrame(score_rows)


def split_tail_by_attachment_elbow(attachment_scores_df: pd.DataFrame, args):
    scores_df = attachment_scores_df.copy().reset_index(drop=True)
    if len(scores_df) == 0:
        summary = {'status': 'empty_tail', 'num_tail': 0, 'elbow_valid': False}
        return scores_df.copy(), scores_df.copy(), summary

    min_segment = int(getattr(args, 'tgb_elbow_min_segment', 3))
    scores_df['attachment_score'] = scores_df['attachment_score'].astype(float)
    sorted_df = scores_df.sort_values('attachment_score', ascending=True).reset_index(drop=True)
    m = int(len(sorted_df))
    valid_group_count = int((scores_df['best_group_id'].astype(int) >= 0).sum())

    if m < max(2 * min_segment, 4) or valid_group_count == 0:
        summary = {
            'status': 'fallback_insufficient_tail_or_groups',
            'num_tail': m,
            'min_segment': int(min_segment),
            'elbow_valid': False,
            'bic_1': None,
            'bic_2': None,
            'break_index': None,
            'threshold': None,
        }
        return sorted_df.copy(), sorted_df.iloc[0:0].copy(), summary

    x = np.arange(m, dtype=float)
    y = sorted_df['attachment_score'].to_numpy(dtype=float)
    sse_1 = _linear_sse(x, y)
    bic_1 = _bic(sse_1, m, q=2)

    best = None
    for break_index in range(min_segment - 1, m - min_segment):
        left = slice(0, break_index + 1)
        right = slice(break_index + 1, m)
        sse_2 = _linear_sse(x[left], y[left]) + _linear_sse(x[right], y[right])
        bic_2 = _bic(sse_2, m, q=4)
        if best is None or bic_2 < best['bic_2']:
            best = {'break_index': int(break_index), 'sse_2': float(sse_2), 'bic_2': float(bic_2)}

    if best is None or not (best['bic_2'] < bic_1):
        summary = {
            'status': 'fallback_no_bic_gain',
            'num_tail': m,
            'min_segment': int(min_segment),
            'elbow_valid': False,
            'sse_1': float(sse_1),
            'bic_1': float(bic_1),
            'bic_2': None if best is None else float(best['bic_2']),
            'break_index': None,
            'threshold': None,
        }
        return sorted_df.copy(), sorted_df.iloc[0:0].copy(), summary

    threshold = float(y[best['break_index']])
    tail_open_df = sorted_df.loc[sorted_df['attachment_score'] <= threshold].copy().reset_index(drop=True)
    tail_attached_df = sorted_df.loc[sorted_df['attachment_score'] > threshold].copy().reset_index(drop=True)
    summary = {
        'status': 'bic_elbow',
        'num_tail': m,
        'min_segment': int(min_segment),
        'elbow_valid': True,
        'sse_1': float(sse_1),
        'sse_2': float(best['sse_2']),
        'bic_1': float(bic_1),
        'bic_2': float(best['bic_2']),
        'break_index': int(best['break_index']),
        'threshold': threshold,
        'num_tail_open': int(len(tail_open_df)),
        'num_tail_attached': int(len(tail_attached_df)),
    }
    return tail_open_df, tail_attached_df, summary
