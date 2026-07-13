from typing import Dict, Optional, Tuple

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


MEMBERSHIP_SCORE_COLUMNS = [
    'best_group_id',
    'best_group_affinity',
    'second_group_id',
    'second_group_affinity',
    'membership_margin',
    'group_membership_threshold',
    'membership_margin_minus_threshold',
    'membership_status',
]

MEMBERSHIP_CALIBRATION_COLUMNS = [
    'group_id',
    'num_clean_members',
    'k_neighbors',
    'num_positive_calibration',
    'num_negative_calibration',
    'membership_threshold',
    'pseudo_tpr',
    'pseudo_fpr',
    'pseudo_youden_j',
    'pseudo_balanced_accuracy',
    'calibration_status',
]


def _membership_embedding_state(grouping_embeddings_payload: Dict):
    sample_indices = [int(value) for value in grouping_embeddings_payload['sample_idx']]
    embeddings = _normalize_rows(grouping_embeddings_payload['embeddings'].float()).cpu()
    return embeddings, {sample_idx: position for position, sample_idx in enumerate(sample_indices)}


def _membership_group_entries(geometry: Dict, sample_positions: Dict[int, int]):
    entries = {}
    for group_id, group in sorted(geometry.get('groups', {}).items()):
        if not bool(group.get('valid_group', False)):
            continue
        sample_indices = [
            int(sample_idx) for sample_idx in group.get('sample_indices', [])
            if int(sample_idx) in sample_positions
        ]
        if len(sample_indices) < 2:
            continue
        entries[int(group_id)] = {
            'group_id': int(group_id),
            'sample_indices': sample_indices,
            'member_positions': torch.as_tensor(
                [sample_positions[sample_idx] for sample_idx in sample_indices],
                dtype=torch.long,
            ),
            'k_neighbors': max(1, int(np.floor(np.sqrt(len(sample_indices))))),
        }
    return entries


def _local_group_affinity(vector: torch.Tensor,
                          entry: Dict,
                          embeddings: torch.Tensor,
                          exclude_sample_idx: Optional[int] = None):
    member_positions = entry['member_positions']
    if exclude_sample_idx is not None:
        keep = torch.as_tensor(
            [sample_idx != int(exclude_sample_idx) for sample_idx in entry['sample_indices']],
            dtype=torch.bool,
        )
        member_positions = member_positions[keep]
    if len(member_positions) == 0:
        return None
    similarities = embeddings[member_positions] @ vector
    k_neighbors = min(int(entry['k_neighbors']), int(len(similarities)))
    return float(torch.topk(similarities, k=k_neighbors, largest=True).values.mean().item())


def _membership_affinities(vector: torch.Tensor,
                            entries: Dict[int, Dict],
                            embeddings: torch.Tensor,
                            own_group_id: Optional[int] = None,
                            sample_idx: Optional[int] = None):
    affinities = {}
    for group_id, entry in entries.items():
        exclude_sample_idx = sample_idx if own_group_id == group_id else None
        affinity = _local_group_affinity(
            vector,
            entry,
            embeddings,
            exclude_sample_idx=exclude_sample_idx,
        )
        if affinity is not None:
            affinities[int(group_id)] = float(affinity)
    return affinities


def _best_membership_groups(affinities: Dict[int, float]):
    ranked = sorted(affinities.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) < 2:
        return None
    best_group_id, best_affinity = ranked[0]
    second_group_id, second_affinity = ranked[1]
    return (
        int(best_group_id),
        float(best_affinity),
        int(second_group_id),
        float(second_affinity),
        float(best_affinity - second_affinity),
    )


def _select_youden_threshold(positive_values, negative_values):
    positives = np.asarray(positive_values, dtype=float)
    negatives = np.asarray(negative_values, dtype=float)
    values = np.concatenate([positives, negatives])
    labels = np.concatenate([np.ones(len(positives), dtype=bool), np.zeros(len(negatives), dtype=bool)])
    order = np.argsort(values)[::-1]
    positive_seen = 0
    negative_seen = 0
    best = None
    index = 0
    while index < len(order):
        threshold = float(values[order[index]])
        end = index
        while end < len(order) and values[order[end]] == threshold:
            if labels[order[end]]:
                positive_seen += 1
            else:
                negative_seen += 1
            end += 1
        tpr = float(positive_seen / len(positives))
        fpr = float(negative_seen / len(negatives))
        candidate = (threshold, tpr, fpr, float(tpr - fpr))
        if best is None or candidate[3] > best[3] or (
            np.isclose(candidate[3], best[3]) and candidate[0] < best[0]
        ):
            best = candidate
        index = end
    threshold, tpr, fpr, youden_j = best
    return {
        'membership_threshold': float(threshold),
        'pseudo_tpr': float(tpr),
        'pseudo_fpr': float(fpr),
        'pseudo_youden_j': float(youden_j),
        'pseudo_balanced_accuracy': float((tpr + (1.0 - fpr)) / 2.0),
    }


def calibrate_h_group_membership(geometry: Dict, grouping_embeddings_payload: Dict):
    embeddings, sample_positions = _membership_embedding_state(grouping_embeddings_payload)
    entries = _membership_group_entries(geometry, sample_positions)
    positive_margins = {group_id: [] for group_id in entries}
    negative_margins = {group_id: [] for group_id in entries}

    for own_group_id, entry in entries.items():
        for sample_idx, position in zip(entry['sample_indices'], entry['member_positions'].tolist()):
            affinities = _membership_affinities(
                embeddings[position],
                entries,
                embeddings,
                own_group_id=int(own_group_id),
                sample_idx=int(sample_idx),
            )
            for group_id, affinity in affinities.items():
                alternatives = [value for other_group_id, value in affinities.items() if other_group_id != group_id]
                if len(alternatives) == 0:
                    continue
                margin = float(affinity - max(alternatives))
                if group_id == own_group_id:
                    positive_margins[group_id].append(margin)
                else:
                    negative_margins[group_id].append(margin)

    calibration = {'groups': {}}
    rows = []
    for group_id, entry in entries.items():
        positives = positive_margins[group_id]
        negatives = negative_margins[group_id]
        row = {
            'group_id': int(group_id),
            'num_clean_members': int(len(entry['sample_indices'])),
            'k_neighbors': int(entry['k_neighbors']),
            'num_positive_calibration': int(len(positives)),
            'num_negative_calibration': int(len(negatives)),
            'membership_threshold': np.nan,
            'pseudo_tpr': np.nan,
            'pseudo_fpr': np.nan,
            'pseudo_youden_j': np.nan,
            'pseudo_balanced_accuracy': np.nan,
            'calibration_status': 'invalid_insufficient_positive_or_negative',
        }
        if len(positives) > 0 and len(negatives) > 0:
            metrics = _select_youden_threshold(positives, negatives)
            row.update(metrics)
            row['calibration_status'] = 'calibrated'
            calibration['groups'][int(group_id)] = {
                'membership_threshold': float(metrics['membership_threshold']),
                'k_neighbors': int(entry['k_neighbors']),
                'num_clean_members': int(len(entry['sample_indices'])),
            }
        rows.append(row)

    for group_id, group in sorted(geometry.get('groups', {}).items()):
        group_id = int(group_id)
        if group_id in entries:
            continue
        rows.append({
            'group_id': group_id,
            'num_clean_members': int(group.get('group_size', 0)),
            'k_neighbors': 0,
            'num_positive_calibration': 0,
            'num_negative_calibration': 0,
            'membership_threshold': np.nan,
            'pseudo_tpr': np.nan,
            'pseudo_fpr': np.nan,
            'pseudo_youden_j': np.nan,
            'pseudo_balanced_accuracy': np.nan,
            'calibration_status': 'invalid_group_size_or_embedding',
        })

    calibration['entries'] = {
        group_id: entries[group_id] for group_id in calibration['groups']
    }
    calibration['embeddings'] = embeddings
    calibration['sample_positions'] = sample_positions
    calibration_summary = {
        'mode': 'h_calibrated',
        'num_geometry_groups': int(len(geometry.get('groups', {}))),
        'num_membership_candidate_groups': int(len(entries)),
        'num_calibrated_groups': int(len(calibration['groups'])),
        'minimum_calibrated_groups': 2,
    }
    return calibration, pd.DataFrame(rows, columns=MEMBERSHIP_CALIBRATION_COLUMNS), calibration_summary


def classify_tail_h_calibrated_membership(tail_df: pd.DataFrame, calibration: Dict):
    scores_df = tail_df.copy().reset_index(drop=True)
    for column in MEMBERSHIP_SCORE_COLUMNS:
        if column not in scores_df.columns:
            scores_df[column] = np.nan
    entries = calibration.get('entries', {})
    thresholds = calibration.get('groups', {})
    has_enough_groups = len(entries) >= 2
    embeddings = calibration['embeddings']
    sample_positions = calibration['sample_positions']
    rows = []

    for _, row in scores_df.iterrows():
        out_row = row.to_dict()
        sample_idx = int(row['sample_idx'])
        position = sample_positions.get(sample_idx)
        vector = None if position is None else embeddings[position]
        if not has_enough_groups:
            out_row.update({
                'best_group_id': -1,
                'best_group_affinity': np.nan,
                'second_group_id': -1,
                'second_group_affinity': np.nan,
                'membership_margin': np.nan,
                'group_membership_threshold': np.nan,
                'membership_margin_minus_threshold': np.nan,
                'membership_status': 'open_insufficient_calibrated_groups',
            })
        elif vector is None:
            out_row.update({
                'best_group_id': -1,
                'best_group_affinity': np.nan,
                'second_group_id': -1,
                'second_group_affinity': np.nan,
                'membership_margin': np.nan,
                'group_membership_threshold': np.nan,
                'membership_margin_minus_threshold': np.nan,
                'membership_status': 'open_missing_embedding',
            })
        else:
            membership = _best_membership_groups(_membership_affinities(vector, entries, embeddings))
            if membership is None:
                out_row.update({
                    'best_group_id': -1,
                    'best_group_affinity': np.nan,
                    'second_group_id': -1,
                    'second_group_affinity': np.nan,
                    'membership_margin': np.nan,
                    'group_membership_threshold': np.nan,
                    'membership_margin_minus_threshold': np.nan,
                    'membership_status': 'open_insufficient_calibrated_groups',
                })
            else:
                best_group_id, best_affinity, second_group_id, second_affinity, margin = membership
                threshold = float(thresholds[best_group_id]['membership_threshold'])
                margin_minus_threshold = float(margin - threshold)
                out_row.update({
                    'best_group_id': int(best_group_id),
                    'best_group_affinity': float(best_affinity),
                    'second_group_id': int(second_group_id),
                    'second_group_affinity': float(second_affinity),
                    'membership_margin': float(margin),
                    'group_membership_threshold': threshold,
                    'membership_margin_minus_threshold': margin_minus_threshold,
                    'membership_status': 'attached' if margin >= threshold else 'open_margin_below_threshold',
                })
        rows.append(out_row)

    if len(rows) == 0:
        return scores_df, scores_df.copy(), scores_df.copy(), {
            'mode': 'h_calibrated',
            'status': 'empty_tail',
            'num_tail': 0,
            'num_tail_open': 0,
            'num_tail_attached': 0,
            'num_calibrated_groups': int(len(entries)),
        }
    scores_df = pd.DataFrame(rows)
    attached_df = scores_df.loc[scores_df['membership_status'] == 'attached'].copy().reset_index(drop=True)
    open_df = scores_df.loc[scores_df['membership_status'] != 'attached'].copy().reset_index(drop=True)
    summary = {
        'mode': 'h_calibrated',
        'status': 'per_sample_membership' if has_enough_groups else 'fallback_insufficient_calibrated_groups',
        'num_tail': int(len(scores_df)),
        'num_tail_open': int(len(open_df)),
        'num_tail_attached': int(len(attached_df)),
        'num_calibrated_groups': int(len(entries)),
        'global_elbow_used': False,
    }
    return scores_df, open_df, attached_df, summary


def build_tail_attachment_plan(tail_df: pd.DataFrame,
                               h_clean_df: pd.DataFrame,
                               grouping_embeddings_payload: Dict,
                               args,
                               membership_mode: str):
    geometry, geometry_summary_df = build_clean_head_geometry(
        h_clean_df,
        grouping_embeddings_payload,
        args,
    )
    if membership_mode == 'h_calibrated':
        calibration, calibration_df, calibration_summary = calibrate_h_group_membership(
            geometry,
            grouping_embeddings_payload,
        )
        scores_df, tail_open_df, tail_attached_df, membership_summary = classify_tail_h_calibrated_membership(
            tail_df,
            calibration,
        )
        membership_summary.update(calibration_summary)
        return {
            'geometry': geometry,
            'conformity_df': pd.DataFrame(columns=['sample_idx', 'sample_key', 'group_id', 'distance', 'conformity']),
            'attachment_scores_df': scores_df,
            'tail_open_df': tail_open_df,
            'tail_attached_df': tail_attached_df,
            'attachment_summary': dict(membership_summary),
            'membership_scores_df': scores_df,
            'membership_calibration_df': calibration_df,
            'membership_summary': membership_summary,
        }
    if membership_mode != 'empirical_conformity':
        raise ValueError('unknown TailGuard-B attachment membership mode: {}'.format(membership_mode))
    conformity_df, scores_df = compute_tail_attachment(
        tail_df,
        geometry,
        grouping_embeddings_payload,
    )
    if len(conformity_df) == 0:
        conformity_df = geometry_summary_df.copy()
    tail_open_df, tail_attached_df, attachment_summary = split_tail_by_attachment_elbow(scores_df, args)
    return {
        'geometry': geometry,
        'conformity_df': conformity_df,
        'attachment_scores_df': scores_df,
        'tail_open_df': tail_open_df,
        'tail_attached_df': tail_attached_df,
        'attachment_summary': attachment_summary,
        'membership_scores_df': None,
        'membership_calibration_df': None,
        'membership_summary': None,
    }
