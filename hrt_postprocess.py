from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from feature_grouping import extract_image_embeddings, fit_mutual_knn_groups
from one_shot_memory import extract_encoder_patch_tokens, compute_memory_patch_scores
from tail_sampler_analysis import build_tail_candidate_partition, compute_selection_metrics


EPS = 1e-12


def _normalize_np(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=EPS, a_max=None)
    return array / norms


def _as_sample_index_set(values) -> set:
    return {int(value) for value in values}


def _patch_feature_mean(patch_tokens: torch.Tensor, keep_mask: np.ndarray = None) -> np.ndarray:
    if keep_mask is None:
        feature = patch_tokens.mean(dim=0)
    else:
        keep_tensor = torch.as_tensor(np.asarray(keep_mask, dtype=bool), device=patch_tokens.device)
        if int(keep_tensor.sum().item()) == 0:
            feature = patch_tokens.mean(dim=0)
        else:
            feature = patch_tokens[keep_tensor].mean(dim=0)
    feature = feature.detach().cpu().numpy().astype(np.float32, copy=False)
    return _normalize_np(feature)[0]


def _build_group_tables(candidate_df: pd.DataFrame, group_assignments: Dict[int, int]) -> Tuple[pd.DataFrame, Dict[int, List[int]]]:
    groups_df = candidate_df.copy()
    groups_df['group_id'] = groups_df['sample_idx'].map({int(k): int(v) for k, v in group_assignments.items()}).astype(int)
    group_sizes = groups_df.groupby('group_id').size().rename('group_size').reset_index()
    groups_df = groups_df.merge(group_sizes, on='group_id', how='left')
    group_to_sample_indices = {
        int(group_id): [int(value) for value in group['sample_idx'].tolist()]
        for group_id, group in groups_df.groupby('group_id')
    }
    return groups_df, group_to_sample_indices


def _retrieve_head_references(
    hrt_embeddings: np.ndarray,
    metadata_df: pd.DataFrame,
    group_to_sample_indices: Dict[int, List[int]],
    head_df: pd.DataFrame,
    top_m: int,
):
    embedding_by_sample = {
        int(sample_idx): hrt_embeddings[index]
        for index, sample_idx in enumerate(metadata_df['sample_idx'].astype(int).tolist())
    }
    head_embeddings = _normalize_np(np.stack([embedding_by_sample[int(sample_idx)] for sample_idx in head_df['sample_idx'].tolist()], axis=0))
    head_sample_indices = head_df['sample_idx'].astype(int).tolist()

    head_reference_rows = []
    group_head_refs = {}

    for group_id, sample_indices in group_to_sample_indices.items():
        group_embeddings = _normalize_np(np.stack([embedding_by_sample[int(sample_idx)] for sample_idx in sample_indices], axis=0))
        group_center = _normalize_np(group_embeddings.mean(axis=0, keepdims=True))[0]

        similarities = head_embeddings @ group_center
        order = np.argsort(-similarities)
        selected_order = order[: min(int(top_m), len(order))]
        refs = []
        for rank, head_position in enumerate(selected_order.tolist(), start=1):
            head_sample_idx = int(head_sample_indices[head_position])
            refs.append(head_sample_idx)
            head_row = head_df.iloc[head_position]
            head_reference_rows.append({
                'group_id': int(group_id),
                'head_sample_idx': head_sample_idx,
                'head_img_path': head_row.get('img_path', None),
                'head_class_name': head_row.get('class_name', None),
                'rank': int(rank),
                'raw_similarity': float(similarities[head_position]),
            })
        group_head_refs[int(group_id)] = refs

    return pd.DataFrame(head_reference_rows), group_head_refs


@torch.no_grad()
def _collect_patch_bank(model, dataloader, device, required_sample_indices: set):
    was_training = model.training
    model.eval()
    patch_bank = {}

    for images, _, meta in dataloader:
        batch_sample_indices = [int(value) for value in meta['sample_idx']]
        keep_positions = [index for index, sample_idx in enumerate(batch_sample_indices) if sample_idx in required_sample_indices]
        if len(keep_positions) == 0:
            continue

        images = images.to(device)
        patch_tokens, spatial_size = extract_encoder_patch_tokens(model, images)
        patch_tokens = patch_tokens.detach().cpu()

        for index in keep_positions:
            sample_idx = int(batch_sample_indices[index])
            patch_bank[sample_idx] = {
                'patch_tokens': patch_tokens[index].contiguous(),
                'spatial_size': tuple(int(v) for v in spatial_size),
                'img_path': meta['img_path'][index] if 'img_path' in meta else None,
                'class_name': meta['class_name'][index] if 'class_name' in meta else None,
                'class_id': int(meta['class_id'][index]) if 'class_id' in meta else None,
            }

    if was_training:
        model.train()
    return patch_bank


def _robust_keep_mask(distances: np.ndarray, trim_lambda: float, min_keep_ratio: float, max_drop_ratio: float) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float32).reshape(-1)
    median_value = float(np.median(distances))
    mad_value = float(np.median(np.abs(distances - median_value)))
    threshold = median_value + float(trim_lambda) * mad_value
    keep_mask = distances <= threshold

    num_patches = int(distances.shape[0])
    min_keep = max(1, int(np.ceil(num_patches * float(min_keep_ratio))))
    max_drop = int(np.floor(num_patches * float(max_drop_ratio)))
    min_keep = max(min_keep, num_patches - max_drop)
    min_keep = min(num_patches, max(1, min_keep))

    if int(keep_mask.sum()) < min_keep:
        keep_indices = np.argsort(distances)[:min_keep]
        keep_mask = np.zeros(num_patches, dtype=bool)
        keep_mask[keep_indices] = True
    return keep_mask


def _largest_component_ratio(mask: np.ndarray, spatial_size: Tuple[int, int]) -> float:
    flat_mask = np.asarray(mask, dtype=bool).reshape(-1)
    if flat_mask.sum() == 0:
        return 0.0
    height, width = int(spatial_size[0]), int(spatial_size[1])
    grid = flat_mask.reshape(height, width)
    visited = np.zeros_like(grid, dtype=bool)
    largest = 0

    for row in range(height):
        for col in range(width):
            if not grid[row, col] or visited[row, col]:
                continue
            stack = [(row, col)]
            visited[row, col] = True
            size = 0
            while stack:
                cur_row, cur_col = stack.pop()
                size += 1
                for next_row, next_col in ((cur_row - 1, cur_col), (cur_row + 1, cur_col), (cur_row, cur_col - 1), (cur_row, cur_col + 1)):
                    if next_row < 0 or next_row >= height or next_col < 0 or next_col >= width:
                        continue
                    if not grid[next_row, next_col] or visited[next_row, next_col]:
                        continue
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))
            largest = max(largest, size)
    return float(largest / max(1, int(flat_mask.sum())))


def _mean_topk_cosine(query: np.ndarray, references: np.ndarray, topk: int) -> float:
    references = _normalize_np(references)
    query = _normalize_np(query.reshape(1, -1))[0]
    similarities = references @ query
    if similarities.size == 0:
        return 0.0
    topk = max(1, min(int(topk), int(similarities.shape[0])))
    values = np.sort(similarities)[-topk:]
    return float(np.mean(values))


def _jaccard_mean(neighbor_sets: List[set]) -> float:
    if len(neighbor_sets) <= 1:
        return 1.0
    overlaps = []
    for index in range(len(neighbor_sets)):
        for other_index in range(index + 1, len(neighbor_sets)):
            union = neighbor_sets[index] | neighbor_sets[other_index]
            if len(union) == 0:
                overlaps.append(1.0)
            else:
                overlaps.append(len(neighbor_sets[index] & neighbor_sets[other_index]) / len(union))
    if len(overlaps) == 0:
        return 1.0
    return float(np.mean(overlaps))


def _pairwise_cosine_mean(vectors: List[np.ndarray]) -> float:
    if len(vectors) <= 1:
        return 1.0
    matrix = _normalize_np(np.stack([np.asarray(vector, dtype=np.float32).reshape(-1) for vector in vectors], axis=0))
    similarity = matrix @ matrix.T
    upper = similarity[np.triu_indices(similarity.shape[0], k=1)]
    if upper.size == 0:
        return 1.0
    return float(np.mean(upper))


@torch.no_grad()
def _build_head_calibration(
    head_df: pd.DataFrame,
    head_embeddings_map: Dict[int, np.ndarray],
    head_patch_bank: Dict[int, Dict],
    top_m: int,
    top_k_mean: int,
    trim_lambda: float,
    min_keep_ratio: float,
    max_drop_ratio: float,
    chunk_size: int,
):
    head_rows = head_df.reset_index(drop=True)
    sample_indices = head_rows['sample_idx'].astype(int).tolist()
    raw_patch_features = {
        int(sample_idx): _patch_feature_mean(head_patch_bank[int(sample_idx)]['patch_tokens'])
        for sample_idx in sample_indices
        if int(sample_idx) in head_patch_bank
    }

    temp_cache = {}
    for sample_idx in sample_indices:
        sample_idx = int(sample_idx)
        if sample_idx not in head_patch_bank:
            continue
        candidate_embedding = head_embeddings_map[int(sample_idx)]
        reference_indices = [int(other_idx) for other_idx in sample_indices if int(other_idx) != int(sample_idx) and int(other_idx) in head_patch_bank]
        if len(reference_indices) == 0:
            continue
        reference_embeddings = _normalize_np(np.stack([head_embeddings_map[idx] for idx in reference_indices], axis=0))
        similarities = reference_embeddings @ _normalize_np(candidate_embedding)[0]
        order = np.argsort(-similarities)[: min(int(top_m), len(similarities))]
        selected_reference_indices = [reference_indices[position] for position in order.tolist()]
        head_memory = torch.cat([head_patch_bank[idx]['patch_tokens'] for idx in selected_reference_indices], dim=0)
        patch_scores = compute_memory_patch_scores(
            head_patch_bank[int(sample_idx)]['patch_tokens'],
            head_memory,
            chunk_size=int(chunk_size),
        ).detach().cpu().numpy()
        keep_mask = _robust_keep_mask(
            patch_scores,
            trim_lambda=trim_lambda,
            min_keep_ratio=min_keep_ratio,
            max_drop_ratio=max_drop_ratio,
        )
        removed_mask = ~keep_mask
        temp_cache[sample_idx] = {
            'reference_indices': selected_reference_indices,
            'raw_patch_feature': raw_patch_features[sample_idx],
            'clean_patch_feature': _patch_feature_mean(head_patch_bank[int(sample_idx)]['patch_tokens'], keep_mask=keep_mask),
            'patch_scores': patch_scores,
            'drop_ratio': float(removed_mask.mean()),
            'keep_ratio': float(keep_mask.mean()),
            'largest_dev_cc_ratio': float(_largest_component_ratio(removed_mask, head_patch_bank[int(sample_idx)]['spatial_size'])),
        }

    rows = []
    for sample_idx, entry in temp_cache.items():
        reference_indices = entry['reference_indices']
        reference_clean_features = np.stack([
            temp_cache[idx]['clean_patch_feature'] if idx in temp_cache else raw_patch_features[idx]
            for idx in reference_indices
        ], axis=0)
        reference_raw_embeddings = np.stack([head_embeddings_map[idx] for idx in reference_indices], axis=0)

        s_raw_cls = _mean_topk_cosine(head_embeddings_map[sample_idx], reference_raw_embeddings, top_k_mean)
        s_raw_patch = _mean_topk_cosine(entry['raw_patch_feature'], reference_clean_features, top_k_mean)
        s_clean = _mean_topk_cosine(entry['clean_patch_feature'], reference_clean_features, top_k_mean)
        neighbor_scores = _normalize_np(reference_clean_features) @ _normalize_np(entry['clean_patch_feature'])[0]
        neighbor_order = np.argsort(-neighbor_scores)[: min(int(top_k_mean), len(neighbor_scores))]
        clean_neighbors = [int(reference_indices[position]) for position in neighbor_order.tolist()]
        rows.append({
            'sample_idx': int(sample_idx),
            's_raw_cls': float(s_raw_cls),
            's_raw_patch': float(s_raw_patch),
            's_clean': float(s_clean),
            'delta': float(s_clean - s_raw_patch),
            'drop_ratio': float(entry['drop_ratio']),
            'keep_ratio': float(entry['keep_ratio']),
            'largest_dev_cc_ratio': float(entry['largest_dev_cc_ratio']),
            'clean_head_neighbor_ids': '|'.join(str(value) for value in clean_neighbors),
        })

    calibration_df = pd.DataFrame(rows)
    return calibration_df, temp_cache


def _build_calibration_summary(calibration_df: pd.DataFrame, sample_scores_df: pd.DataFrame) -> Dict[str, float]:
    source_df = calibration_df if len(calibration_df) > 0 else sample_scores_df
    if len(source_df) == 0:
        return {
            'head_patch_raw_similarity_q10': 0.0,
            'head_patch_raw_similarity_q25': 0.0,
            'head_clean_similarity_q10': 0.0,
            'head_clean_similarity_q25': 0.0,
            'head_gain_q50': 0.0,
            'head_gain_q75': 0.0,
            'head_gain_q90': 0.0,
            'head_drop_ratio_q75': 0.0,
            'head_drop_ratio_q90': 0.0,
            'head_largest_dev_cc_ratio_q75': 0.0,
            'head_largest_dev_cc_ratio_q90': 0.0,
            'neighbor_consistency_q50': 0.35,
        }

    if 'clean_head_neighbor_ids' in source_df.columns and len(source_df) > 0:
        calibration_neighbor_consistency = _jaccard_mean([
            {int(value) for value in str(text).split('|') if str(value).strip()}
            for text in source_df['clean_head_neighbor_ids'].fillna('')
        ])
    else:
        calibration_neighbor_consistency = 0.35

    s_raw_col = 's_raw_patch' if 's_raw_patch' in source_df.columns else 's_raw'
    return {
        'head_patch_raw_similarity_q10': float(source_df[s_raw_col].quantile(0.10)),
        'head_patch_raw_similarity_q25': float(source_df[s_raw_col].quantile(0.25)),
        'head_clean_similarity_q10': float(source_df['s_clean'].quantile(0.10)),
        'head_clean_similarity_q25': float(source_df['s_clean'].quantile(0.25)),
        'head_gain_q50': float(source_df['delta'].quantile(0.50)),
        'head_gain_q75': float(source_df['delta'].quantile(0.75)),
        'head_gain_q90': float(source_df['delta'].quantile(0.90)),
        'head_drop_ratio_q75': float(source_df['drop_ratio'].quantile(0.75)),
        'head_drop_ratio_q90': float(source_df['drop_ratio'].quantile(0.90)),
        'head_largest_dev_cc_ratio_q75': float(source_df['largest_dev_cc_ratio'].quantile(0.75)),
        'head_largest_dev_cc_ratio_q90': float(source_df['largest_dev_cc_ratio'].quantile(0.90)),
        'neighbor_consistency_q50': float(calibration_neighbor_consistency),
    }


def _compute_group_decisions(sample_scores_df: pd.DataFrame, calibration: Dict[str, float], deviation_maps: Dict[int, torch.Tensor]) -> pd.DataFrame:
    columns = [
        'group_id',
        'group_size',
        'median_s_raw_cls',
        'median_s_raw_patch',
        'median_s_clean',
        'median_delta',
        'median_drop_ratio',
        'median_largest_dev_cc_ratio',
        'neighbor_consistency',
        'deviation_map_consistency',
        'head_like',
        'strong_gain',
        'localized_pattern',
        'consistent_reattachment',
        'decision_reason',
        'decision',
    ]
    rows = []
    for group_id, group_df in sample_scores_df.groupby('group_id'):
        median_s_raw_cls = float(group_df['s_raw_cls'].median()) if 's_raw_cls' in group_df.columns else 0.0
        median_s_raw_patch = float(group_df['s_raw'].median())
        median_s_clean = float(group_df['s_clean'].median())
        median_delta = float(group_df['delta'].median())
        median_drop_ratio = float(group_df['drop_ratio'].median())
        median_cc = float(group_df['largest_dev_cc_ratio'].median())
        neighbor_consistency = _jaccard_mean([
            {int(value) for value in str(text).split('|') if str(value).strip()}
            for text in group_df['clean_head_neighbor_ids'].fillna('')
        ])
        map_vectors = [
            deviation_maps[int(sample_idx)].detach().cpu().numpy().reshape(-1)
            for sample_idx in group_df['sample_idx'].astype(int).tolist()
            if int(sample_idx) in deviation_maps
        ]
        deviation_map_consistency = _pairwise_cosine_mean(map_vectors)

        head_like = median_s_clean >= float(calibration['head_clean_similarity_q10'])
        strong_gain = median_delta >= float(calibration['head_gain_q75'])
        moderate_gain = median_delta >= float(calibration['head_gain_q50'])
        localized_pattern = (
            median_drop_ratio <= float(calibration['head_drop_ratio_q90']) + 0.10
            and median_cc >= max(0.25, float(calibration['head_largest_dev_cc_ratio_q75']))
        )
        consistent_reattachment = (
            neighbor_consistency >= max(0.35, float(calibration['neighbor_consistency_q50']))
            or deviation_map_consistency >= 0.60
        )

        if head_like and strong_gain and (localized_pattern or consistent_reattachment):
            decision = 'noise_subcluster'
            decision_reason = 'head_like_after_cleaning_with_structured_gain'
        elif median_s_clean < float(calibration['head_clean_similarity_q25']) and not moderate_gain:
            decision = 'clean_tail'
            decision_reason = 'still_far_from_head_after_cleaning'
        elif median_s_clean < float(calibration['head_clean_similarity_q10']) and median_delta < float(calibration['head_gain_q75']):
            decision = 'clean_tail'
            decision_reason = 'low_reattachment_and_limited_recovery_gain'
        else:
            decision = 'uncertain_tail'
            decision_reason = 'mixed_or_borderline_group'

        rows.append({
            'group_id': int(group_id),
            'group_size': int(len(group_df)),
            'median_s_raw_cls': median_s_raw_cls,
            'median_s_raw_patch': median_s_raw_patch,
            'median_s_clean': median_s_clean,
            'median_delta': median_delta,
            'median_drop_ratio': median_drop_ratio,
            'median_largest_dev_cc_ratio': median_cc,
            'neighbor_consistency': float(neighbor_consistency),
            'deviation_map_consistency': float(deviation_map_consistency),
            'head_like': int(head_like),
            'strong_gain': int(strong_gain),
            'localized_pattern': int(localized_pattern),
            'consistent_reattachment': int(consistent_reattachment),
            'decision_reason': decision_reason,
            'decision': decision,
        })
    return pd.DataFrame(rows, columns=columns)


def build_final_hrt_selection(details_df: pd.DataFrame, sample_scores_df: pd.DataFrame):
    required_columns = {'sample_idx', 'is_selected', 'is_gt_tail'}
    missing_columns = required_columns.difference(details_df.columns)
    if missing_columns:
        raise ValueError('details_df missing required columns: {}'.format(sorted(missing_columns)))
    if details_df['sample_idx'].duplicated().any():
        raise ValueError('details_df sample_idx values must be unique')
    if len(sample_scores_df) > 0 and sample_scores_df['sample_idx'].duplicated().any():
        raise ValueError('sample_scores_df sample_idx values must be unique')

    merge_columns = [
        'sample_idx',
        'group_id',
        'group_size',
        's_raw_cls',
        's_raw',
        's_clean',
        'delta',
        'drop_ratio',
        'keep_ratio',
        'largest_dev_cc_ratio',
        'clean_head_neighbor_ids',
        'decision',
    ]
    available_columns = [column for column in merge_columns if column in sample_scores_df.columns]
    if len(sample_scores_df) == 0:
        decision_df = pd.DataFrame(columns=available_columns)
    else:
        decision_df = sample_scores_df[available_columns].copy()
    if 'decision' in decision_df.columns:
        decision_df = decision_df.rename(columns={'decision': 'hrt_decision'})
    else:
        decision_df['hrt_decision'] = pd.Series(dtype=object)

    final_selection_df = details_df.copy()
    final_selection_df['sample_idx'] = final_selection_df['sample_idx'].astype(int)
    final_selection_df = final_selection_df.merge(
        decision_df,
        on='sample_idx',
        how='left',
        validate='one_to_one',
    )
    if len(final_selection_df) != len(details_df):
        raise RuntimeError('HRT final selection merge changed sample count')

    final_selection_df['is_selected_before_hrt'] = final_selection_df['is_selected'].astype(int)
    candidate_mask = final_selection_df['is_selected_before_hrt'] == 1
    missing_candidate_decision = candidate_mask & final_selection_df['hrt_decision'].isna()
    final_selection_df.loc[missing_candidate_decision, 'hrt_decision'] = 'uncertain_tail'
    final_selection_df.loc[~candidate_mask, 'hrt_decision'] = 'reliable_head'
    final_selection_df['hrt_role'] = final_selection_df['hrt_decision']

    noise_mask = candidate_mask & (final_selection_df['hrt_decision'] == 'noise_subcluster')
    final_selection_df['is_selected_after_hrt'] = candidate_mask.astype(int)
    final_selection_df.loc[noise_mask, 'is_selected_after_hrt'] = 0
    final_selection_df['hrt_removed'] = noise_mask.astype(int)

    before_metrics = compute_selection_metrics(
        final_selection_df['is_selected_before_hrt'].to_numpy(),
        final_selection_df['is_gt_tail'].astype(int).to_numpy(),
    )
    after_metrics = compute_selection_metrics(
        final_selection_df['is_selected_after_hrt'].to_numpy(),
        final_selection_df['is_gt_tail'].astype(int).to_numpy(),
    )
    processing_summary = {
        'before_selection_metrics': before_metrics,
        'after_selection_metrics': after_metrics,
        'num_selected_before_hrt': int(final_selection_df['is_selected_before_hrt'].sum()),
        'num_selected_after_hrt': int(final_selection_df['is_selected_after_hrt'].sum()),
        'num_removed_by_hrt': int(final_selection_df['hrt_removed'].sum()),
        'num_candidates_defaulted_to_uncertain': int(missing_candidate_decision.sum()),
        'num_reliable_head_samples': int((final_selection_df['hrt_role'] == 'reliable_head').sum()),
    }
    return final_selection_df, processing_summary


@torch.no_grad()
def run_hrt_postprocess(model, dataloader, device, metadata_df: pd.DataFrame, details_df: pd.DataFrame, args):
    candidate_df, head_df = build_tail_candidate_partition(details_df)
    if len(candidate_df) == 0:
        raise ValueError('HRT requires at least one TailSampler candidate')
    if len(head_df) == 0:
        raise ValueError('HRT requires at least one head reference sample')

    hrt_embeddings, hrt_metadata_df = extract_image_embeddings(
        model,
        dataloader,
        device,
        embedding_source=getattr(args, 'hrt_embedding_source', 'encoder_cls'),
    )
    if len(hrt_metadata_df) != len(metadata_df):
        raise ValueError('HRT metadata size mismatch with base metadata')

    candidate_sample_indices = candidate_df['sample_idx'].astype(int).tolist()
    candidate_positions = hrt_metadata_df['sample_idx'].astype(int).isin(candidate_sample_indices).to_numpy()
    candidate_embeddings = hrt_embeddings[candidate_positions]
    candidate_sample_keys = hrt_metadata_df.loc[candidate_positions, 'sample_idx'].astype(int).tolist()

    group_assignments, grouping_info = fit_mutual_knn_groups(
        candidate_embeddings,
        candidate_sample_keys,
        k=int(getattr(args, 'hrt_group_k', 5)),
        min_similarity=getattr(args, 'hrt_group_min_similarity', None),
        min_group_size=int(getattr(args, 'hrt_group_min_size', 1)),
    )
    candidate_groups_df, group_to_sample_indices = _build_group_tables(candidate_df, group_assignments)

    head_references_df, group_head_refs = _retrieve_head_references(
        hrt_embeddings=hrt_embeddings,
        metadata_df=hrt_metadata_df,
        group_to_sample_indices=group_to_sample_indices,
        head_df=head_df,
        top_m=int(getattr(args, 'hrt_head_topm', 20)),
    )

    required_sample_indices = _as_sample_index_set(candidate_groups_df['sample_idx'].tolist())
    required_sample_indices.update(int(value) for value in head_df['sample_idx'].astype(int).tolist())
    for reference_indices in group_head_refs.values():
        required_sample_indices.update(int(value) for value in reference_indices)
    patch_bank = _collect_patch_bank(model, dataloader, device, required_sample_indices=required_sample_indices)

    embedding_map = {
        int(sample_idx): hrt_embeddings[index]
        for index, sample_idx in enumerate(hrt_metadata_df['sample_idx'].astype(int).tolist())
    }

    sample_rows = []
    deviation_maps = {}
    top_k_mean = int(getattr(args, 'hrt_head_topk_mean', 5))
    trim_lambda = float(getattr(args, 'hrt_trim_lambda', 2.5))
    min_keep_ratio = float(getattr(args, 'hrt_min_keep_ratio', 0.70))
    max_drop_ratio = float(getattr(args, 'hrt_max_drop_ratio', 0.30))
    chunk_size = int(getattr(args, 'hrt_chunk_size', 4096))

    head_embeddings_map = {int(sample_idx): embedding_map[int(sample_idx)] for sample_idx in head_df['sample_idx'].astype(int).tolist()}
    head_patch_bank = {int(sample_idx): patch_bank[int(sample_idx)] for sample_idx in head_df['sample_idx'].astype(int).tolist() if int(sample_idx) in patch_bank}
    calibration_df, head_feature_cache = _build_head_calibration(
        head_df=head_df,
        head_embeddings_map=head_embeddings_map,
        head_patch_bank=head_patch_bank,
        top_m=int(getattr(args, 'hrt_head_topm', 20)),
        top_k_mean=top_k_mean,
        trim_lambda=trim_lambda,
        min_keep_ratio=min_keep_ratio,
        max_drop_ratio=max_drop_ratio,
        chunk_size=chunk_size,
    )

    for group_id, sample_indices in group_to_sample_indices.items():
        reference_indices = [int(value) for value in group_head_refs[int(group_id)] if int(value) in patch_bank]
        if len(reference_indices) == 0:
            continue
        head_memory = torch.cat([patch_bank[idx]['patch_tokens'] for idx in reference_indices], dim=0)
        head_reference_clean_features = np.stack([
            head_feature_cache[idx]['clean_patch_feature'] if idx in head_feature_cache else _patch_feature_mean(patch_bank[idx]['patch_tokens'])
            for idx in reference_indices
        ], axis=0)
        raw_reference_embeddings = np.stack([embedding_map[idx] for idx in reference_indices], axis=0)

        for sample_idx in sample_indices:
            sample_idx = int(sample_idx)
            sample_entry = patch_bank[sample_idx]
            patch_scores = compute_memory_patch_scores(
                sample_entry['patch_tokens'],
                head_memory,
                chunk_size=chunk_size,
            ).detach().cpu().numpy()
            keep_mask = _robust_keep_mask(
                patch_scores,
                trim_lambda=trim_lambda,
                min_keep_ratio=min_keep_ratio,
                max_drop_ratio=max_drop_ratio,
            )
            removed_mask = ~keep_mask
            raw_patch_feature = _patch_feature_mean(sample_entry['patch_tokens'])
            clean_feature = _patch_feature_mean(sample_entry['patch_tokens'], keep_mask=keep_mask)
            s_raw_cls = _mean_topk_cosine(embedding_map[sample_idx], raw_reference_embeddings, top_k_mean)
            s_raw_patch = _mean_topk_cosine(raw_patch_feature, head_reference_clean_features, top_k_mean)
            s_clean = _mean_topk_cosine(clean_feature, head_reference_clean_features, top_k_mean)
            clean_neighbor_scores = _normalize_np(head_reference_clean_features) @ _normalize_np(clean_feature)[0]
            clean_neighbor_order = np.argsort(-clean_neighbor_scores)[: min(int(top_k_mean), len(clean_neighbor_scores))]
            clean_neighbors = [int(reference_indices[position]) for position in clean_neighbor_order.tolist()]
            deviation_maps[sample_idx] = torch.from_numpy(patch_scores.reshape(sample_entry['spatial_size']).astype(np.float32))

            detail_row = candidate_groups_df[candidate_groups_df['sample_idx'].astype(int) == sample_idx].iloc[0]
            sample_rows.append({
                'sample_idx': int(sample_idx),
                'img_path': detail_row.get('img_path', None),
                'class_name': detail_row.get('class_name', None),
                'class_id': int(detail_row['class_id']) if 'class_id' in detail_row and pd.notna(detail_row['class_id']) else None,
                'group_id': int(group_id),
                'group_size': int(detail_row['group_size']),
                'pred_class_size': float(detail_row['pred_class_size']),
                'tail_score': float(detail_row['tail_score']),
                's_raw_cls': float(s_raw_cls),
                's_raw': float(s_raw_patch),
                's_clean': float(s_clean),
                'delta': float(s_clean - s_raw_patch),
                'drop_ratio': float(removed_mask.mean()),
                'keep_ratio': float(keep_mask.mean()),
                'largest_dev_cc_ratio': float(_largest_component_ratio(removed_mask, sample_entry['spatial_size'])),
                'clean_head_neighbor_ids': '|'.join(str(value) for value in clean_neighbors),
            })

    sample_scores_df = pd.DataFrame(sample_rows)
    calibration = _build_calibration_summary(calibration_df, sample_scores_df)
    group_decisions_df = _compute_group_decisions(sample_scores_df, calibration, deviation_maps)
    sample_scores_df = sample_scores_df.merge(group_decisions_df[['group_id', 'decision']], on='group_id', how='left')
    final_selection_df, processing_summary = build_final_hrt_selection(details_df, sample_scores_df)

    summary = {
        'num_tail_candidates': int(len(candidate_df)),
        'num_head_samples': int(len(head_df)),
        'num_candidate_groups': int(candidate_groups_df['group_id'].nunique()) if len(candidate_groups_df) > 0 else 0,
        'num_noise_subcluster_groups': int((group_decisions_df['decision'] == 'noise_subcluster').sum()) if len(group_decisions_df) > 0 else 0,
        'num_clean_tail_groups': int((group_decisions_df['decision'] == 'clean_tail').sum()) if len(group_decisions_df) > 0 else 0,
        'num_uncertain_tail_groups': int((group_decisions_df['decision'] == 'uncertain_tail').sum()) if len(group_decisions_df) > 0 else 0,
        'num_noise_subcluster_samples': int((sample_scores_df['decision'] == 'noise_subcluster').sum()) if len(sample_scores_df) > 0 else 0,
        'num_clean_tail_samples': int((sample_scores_df['decision'] == 'clean_tail').sum()) if len(sample_scores_df) > 0 else 0,
        'num_uncertain_tail_samples': int((sample_scores_df['decision'] == 'uncertain_tail').sum()) if len(sample_scores_df) > 0 else 0,
        'grouping_info': grouping_info,
        'calibration': calibration,
        **processing_summary,
    }

    return {
        'summary': summary,
        'candidate_groups_df': candidate_groups_df,
        'head_references_df': head_references_df,
        'sample_scores_df': sample_scores_df,
        'group_decisions_df': group_decisions_df,
        'final_selection_df': final_selection_df,
        'deviation_maps': deviation_maps,
        'patch_bank': patch_bank,
        'calibration_df': calibration_df,
    }
