"""Preparation helpers for the deprecated TGM method."""

import json
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from feature_grouping import extract_image_embeddings, fit_latent_groups
from tail_sampler import build_tail_sampler
from tail_sampler_analysis import run_tail_sampler_analysis, save_tail_sampler_artifacts
from tgm_artifacts import save_tgm_prepare_artifacts


METHOD_METADATA_COLUMNS = [
    'sample_idx',
    'sample_key',
    'img_path',
    'base_idx',
    'tail_candidate',
    'pred_class_size',
    'tail_score',
    'group_id',
    'group_size',
]


GROUP_ASSIGNMENT_COLUMNS = [
    'sample_key',
    'sample_idx',
    'img_path',
    'base_idx',
    'group_id',
    'group_size',
]


CANDIDATE_COLUMNS = [
    'sample_idx',
    'sample_key',
    'img_path',
    'tail_candidate',
    'pred_class_size',
    'tail_score',
]


DEFAULT_K_CANDIDATES = (8, 10, 12, 15, 20)


def _parse_k_candidates(values):
    if values is None:
        return DEFAULT_K_CANDIDATES
    if isinstance(values, str):
        return tuple(int(part.strip()) for part in values.split(',') if part.strip())
    return tuple(int(value) for value in values)


def _normalize_rows(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.float()
    norms = tensor.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return tensor / norms


def _extract_tail_candidates(embeddings: np.ndarray, metadata_df: pd.DataFrame, args):
    sampler, sampler_name = build_tail_sampler(
        sampler_type=getattr(args, 'tailsampler_type', 'adaptive'),
        threshold_type=getattr(args, 'tailsampler_th_type', None),
        vote_type=getattr(args, 'tailsampler_vote_type', None),
        percentile=float(getattr(args, 'tailsampler_percentile', 0.15)),
    )
    features = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))
    _, tail_indices, pred_class_sizes = sampler.run(features, return_class_sizes=True)

    tail_mask = np.zeros(len(metadata_df), dtype=np.int64)
    tail_mask[tail_indices.detach().cpu().numpy().astype(int)] = 1
    pred_class_sizes = pred_class_sizes.detach().cpu().numpy().astype(np.float32)
    tail_scores = 1.0 / np.maximum(pred_class_sizes, 1.0)

    candidates_df = metadata_df[['sample_idx', 'sample_key', 'img_path']].copy()
    candidates_df['tail_candidate'] = tail_mask.astype(int)
    candidates_df['pred_class_size'] = pred_class_sizes.astype(float)
    candidates_df['tail_score'] = tail_scores.astype(float)
    return candidates_df[CANDIDATE_COLUMNS].copy(), sampler_name


def _build_group_assignments(embeddings: np.ndarray, metadata_df: pd.DataFrame, args):
    group_assignments, group_info = fit_latent_groups(
        embeddings,
        metadata_df['sample_key'].tolist(),
        method=getattr(args, 'gbps_grouping_method', 'kmeans'),
        num_groups=getattr(args, 'gbps_num_groups', None),
        auto_k=bool(getattr(args, 'gbps_auto_k', False)),
        k_candidates=_parse_k_candidates(getattr(args, 'gbps_k_candidates', DEFAULT_K_CANDIDATES)),
        pca_dim=getattr(args, 'gbps_pca_dim', 64),
        random_state=0,
    )
    group_df = metadata_df[['sample_key', 'sample_idx', 'img_path', 'base_idx']].copy()
    group_df['group_id'] = group_df['sample_key'].map({int(key): int(value) for key, value in group_assignments.items()})
    group_sizes = group_df.groupby('group_id').size().rename('group_size').reset_index()
    group_df = group_df.merge(group_sizes, on='group_id', how='left')
    group_df['group_id'] = group_df['group_id'].astype(int)
    group_df['group_size'] = group_df['group_size'].astype(int)
    return group_df[GROUP_ASSIGNMENT_COLUMNS].copy(), group_info


@torch.no_grad()
def prepare_tgm_training_metadata(model,
                                        train_eval_dataloader,
                                        device,
                                        output_dir: Optional[str],
                                        args):
    tail_embedding_source = getattr(args, 'tgm_tail_embedding_source', None) or getattr(args, 'tailsampler_embedding_source', 'encoder')
    group_embedding_source = getattr(args, 'gbps_embedding_source', 'encoder')

    tail_embeddings, metadata_df = extract_image_embeddings(
        model,
        train_eval_dataloader,
        device,
        embedding_source=tail_embedding_source,
    )
    metadata_df = metadata_df.copy()
    metadata_df['sample_idx'] = metadata_df['sample_idx'].astype(int)
    metadata_df['sample_key'] = metadata_df['sample_key'].astype(int)
    metadata_df['base_idx'] = metadata_df['base_idx'].astype(int)

    if len(metadata_df) == 0:
        raise ValueError('tgm prepare requires non-empty train metadata')

    if group_embedding_source == tail_embedding_source:
        group_embeddings = tail_embeddings
    else:
        group_embeddings, group_metadata_df = extract_image_embeddings(
            model,
            train_eval_dataloader,
            device,
            embedding_source=group_embedding_source,
        )
        group_metadata_df = group_metadata_df.sort_values('sample_idx').reset_index(drop=True)
        expected_metadata_df = metadata_df.sort_values('sample_idx').reset_index(drop=True)
        if group_metadata_df['sample_idx'].tolist() != expected_metadata_df['sample_idx'].tolist():
            raise ValueError('group embedding metadata mismatch during tgm prepare')

    candidates_df, sampler_name = _extract_tail_candidates(tail_embeddings, metadata_df, args)
    group_assignments_df, group_info = _build_group_assignments(group_embeddings, metadata_df, args)

    full_metadata_df = metadata_df.merge(candidates_df, on=['sample_idx', 'sample_key', 'img_path'], how='left', validate='one_to_one')
    full_metadata_df = full_metadata_df.merge(group_assignments_df, on=['sample_idx', 'sample_key', 'img_path', 'base_idx'], how='left', validate='one_to_one')
    full_metadata_df['tail_candidate'] = full_metadata_df['tail_candidate'].fillna(0).astype(int)
    full_metadata_df['pred_class_size'] = full_metadata_df['pred_class_size'].astype(float)
    full_metadata_df['tail_score'] = full_metadata_df['tail_score'].astype(float)
    full_metadata_df['group_id'] = full_metadata_df['group_id'].astype(int)
    full_metadata_df['group_size'] = full_metadata_df['group_size'].astype(int)

    train_metadata_df = full_metadata_df[METHOD_METADATA_COLUMNS].copy()
    train_metadata_df = train_metadata_df.sort_values('sample_idx').reset_index(drop=True)
    candidates_df = candidates_df.sort_values('sample_idx').reset_index(drop=True)
    group_assignments_df = group_assignments_df.sort_values('sample_idx').reset_index(drop=True)
    full_metadata_df = full_metadata_df.sort_values('sample_idx').reset_index(drop=True)

    cls_embeddings, cls_metadata_df = extract_image_embeddings(
        model,
        train_eval_dataloader,
        device,
        embedding_source='encoder_cls',
    )
    cls_metadata_df = cls_metadata_df.sort_values('sample_idx').reset_index(drop=True)
    if cls_metadata_df['sample_idx'].tolist() != train_metadata_df['sample_idx'].tolist():
        raise ValueError('cls embedding metadata mismatch during tgm prepare')
    cls_tensor = _normalize_rows(torch.from_numpy(np.asarray(cls_embeddings, dtype=np.float32))).cpu()
    cls_embeddings_payload = {
        'sample_idx': cls_metadata_df['sample_idx'].astype(int).tolist(),
        'img_path': cls_metadata_df['img_path'].astype(str).tolist(),
        'embeddings': cls_tensor,
        'embedding_source': 'encoder_cls',
    }

    analysis_saved = None
    analysis_summary = None
    if bool(getattr(args, 'tgm_save_tailsampler_analysis', True)):
        if getattr(args, 'tailsampler_dataset_name', None) is None:
            setattr(args, 'tailsampler_dataset_name', os.path.basename(os.path.normpath(str(getattr(args, 'data_path', '')))))
        analysis_summary, analysis_details_df = run_tail_sampler_analysis(tail_embeddings, full_metadata_df, args)
        if output_dir is not None:
            analysis_output_dir = os.path.join(output_dir, 'tail_sampler_analysis_only')
            analysis_saved = save_tail_sampler_artifacts(
                analysis_output_dir,
                analysis_summary,
                analysis_details_df,
                metadata={
                    'analysis_only': True,
                    'not_used_by_tgm_decision_logic': True,
                    'tail_embedding_source': tail_embedding_source,
                    'gt_mode': getattr(args, 'tailsampler_gt_mode', 'dataset_rule'),
                    'dataset_name': getattr(args, 'tailsampler_dataset_name', None),
                },
                save_details=bool(getattr(args, 'tgm_save_tailsampler_analysis_details', True)),
            )

    prepare_summary = {
        'num_samples': int(len(train_metadata_df)),
        'num_tail_candidates': int(train_metadata_df['tail_candidate'].sum()),
        'tail_candidate_ratio': float(train_metadata_df['tail_candidate'].mean()),
        'num_groups': int(train_metadata_df['group_id'].nunique()),
        'tail_sampler_type': getattr(args, 'tailsampler_type', 'adaptive'),
        'tail_sampler_impl': sampler_name,
        'tail_embedding_source': tail_embedding_source,
        'group_embedding_source': group_embedding_source,
        'group_info': group_info,
        'tail_sampler_analysis_only_summary': analysis_summary,
        'tail_sampler_analysis_only_artifacts': analysis_saved,
    }

    saved = None
    if output_dir is not None:
        metadata = {
            'tail_sampler_type': getattr(args, 'tailsampler_type', 'adaptive'),
            'tailsampler_th_type': getattr(args, 'tailsampler_th_type', None),
            'tailsampler_vote_type': getattr(args, 'tailsampler_vote_type', None),
            'tailsampler_percentile': float(getattr(args, 'tailsampler_percentile', 0.15)),
            'tail_embedding_source': tail_embedding_source,
            'group_embedding_source': group_embedding_source,
            'grouping_method': getattr(args, 'gbps_grouping_method', 'kmeans'),
            'gbps_num_groups': getattr(args, 'gbps_num_groups', None),
            'gbps_auto_k': bool(getattr(args, 'gbps_auto_k', False)),
            'gbps_k_candidates': list(_parse_k_candidates(getattr(args, 'gbps_k_candidates', DEFAULT_K_CANDIDATES))),
            'gbps_pca_dim': getattr(args, 'gbps_pca_dim', 64),
            'summary': prepare_summary,
        }
        saved = save_tgm_prepare_artifacts(
            output_dir,
            candidates_df,
            group_assignments_df,
            train_metadata_df,
            metadata=metadata,
            cls_embeddings_payload=cls_embeddings_payload if bool(getattr(args, 'tgm_save_cls_embeddings', False)) else None,
            analysis_details_df=None,
        )
        with open(os.path.join(output_dir, 'tgm_group_info.json'), 'w', encoding='utf-8') as file:
            json.dump(group_info, file, indent=2, ensure_ascii=False)

    return {
        'candidates_df': candidates_df,
        'group_assignments_df': group_assignments_df,
        'train_metadata_df': train_metadata_df,
        'full_metadata_df': full_metadata_df,
        'cls_embeddings_payload': cls_embeddings_payload,
        'group_info': group_info,
        'summary': prepare_summary,
        'saved': saved,
    }
