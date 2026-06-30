import random
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augment import build_view_transforms
from .calibrator import build_assignment_calibration, build_default_calibration
from .cluster import build_cluster_summary, build_prototype_summary, build_result_table
from .data import MultiViewImagePathDataset, collate_multiview, discover_image_paths
from .discovery import discover_semantic_prototypes
from .encoder import SiglipImageEncoder
from .features import aggregate_multiview, compute_view_stability, stack_view_embeddings
from .io_utils import ensure_dir, save_features_npz, save_prototype_npz, save_result_csv, save_summary_json


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _feature_paths(config) -> Dict[str, str]:
    return {
        'result_csv': '{}/{}'.format(config.output.output_dir, config.output.result_csv),
        'summary_json': '{}/{}'.format(config.output.output_dir, config.output.summary_json),
        'features_npz': '{}/{}'.format(config.output.output_dir, config.output.features_npz),
        'prototype_npz': '{}/{}'.format(config.output.output_dir, config.output.prototype_npz),
    }


def _extract_feature_bundle(config):
    image_paths = discover_image_paths(
        config.data.image_root,
        config.data.image_extensions,
        recursive=bool(config.data.recursive),
        split=str(config.data.split),
    )
    transforms = build_view_transforms(config)
    dataset = MultiViewImagePathDataset(image_paths, transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=int(config.encoder.batch_size),
        shuffle=False,
        num_workers=int(config.encoder.num_workers),
        drop_last=False,
        collate_fn=collate_multiview,
    )
    encoder = SiglipImageEncoder(
        model_name=str(config.encoder.model_name),
        device=str(config.encoder.device),
        precision=str(config.encoder.precision),
    )

    all_embeddings = []
    all_view_embeddings = []
    all_view_stability = []
    all_sample_idx = []
    all_img_path = []

    for view_batches, meta in dataloader:
        encoded_views = [encoder.encode_pil_batch(images) for images in view_batches]
        batch_embedding = aggregate_multiview(encoded_views, normalize=bool(config.encoder.normalize)).cpu()
        batch_view_embeddings = stack_view_embeddings(encoded_views, normalize=bool(config.encoder.normalize)).cpu()
        batch_view_stability = compute_view_stability(encoded_views).cpu()

        all_embeddings.append(batch_embedding)
        all_view_embeddings.append(batch_view_embeddings)
        all_view_stability.append(batch_view_stability)
        all_sample_idx.extend([int(item) for item in meta['sample_idx']])
        all_img_path.extend([str(item) for item in meta['img_path']])

    embeddings = torch.cat(all_embeddings, dim=0).numpy().astype(np.float32)
    view_embeddings = torch.cat(all_view_embeddings, dim=0).numpy().astype(np.float32)
    view_stability = torch.cat(all_view_stability, dim=0).numpy().astype(np.float32)

    return {
        'image_paths': np.asarray(all_img_path),
        'sample_idx': np.asarray(all_sample_idx, dtype=np.int64),
        'embeddings': embeddings,
        'view_embeddings': view_embeddings,
        'view_stability': view_stability,
    }


def _load_feature_bundle(path: str):
    loaded = np.load(path, allow_pickle=False)
    return {
        'image_paths': loaded['image_paths'].astype(str),
        'sample_idx': loaded['sample_idx'].astype(np.int64) if 'sample_idx' in loaded.files else np.arange(len(loaded['image_paths']), dtype=np.int64),
        'embeddings': loaded['embeddings'].astype(np.float32),
        'view_embeddings': loaded['view_embeddings'].astype(np.float32),
        'view_stability': loaded['view_stability'].astype(np.float32),
        'labels': loaded['labels'].astype(np.int64) if 'labels' in loaded.files else None,
        'prototype_ids': loaded['prototype_ids'].astype(np.int64) if 'prototype_ids' in loaded.files else None,
        'assignment_scores': loaded['assignment_scores'].astype(np.float32) if 'assignment_scores' in loaded.files else None,
        'assignment_margins': loaded['assignment_margins'].astype(np.float32) if 'assignment_margins' in loaded.files else None,
        'assignment_thresholds': loaded['assignment_thresholds'].astype(np.float32) if 'assignment_thresholds' in loaded.files else None,
    }


def _run_extract_stage(config):
    feature_bundle = _extract_feature_bundle(config)
    paths = _feature_paths(config)
    save_features_npz(paths['features_npz'], feature_bundle)
    return {
        'num_samples': int(len(feature_bundle['image_paths'])),
        'output_dir': str(config.output.output_dir),
        'features_path': paths['features_npz'],
    }


def _run_cluster_stage(config):
    paths = _feature_paths(config)
    feature_bundle = _load_feature_bundle(paths['features_npz'])
    if bool(config.calibration.enabled):
        calibration = build_assignment_calibration(
            view_embeddings=feature_bundle['view_embeddings'],
            embeddings=feature_bundle['embeddings'],
            num_bins=int(config.calibration.num_bins),
            num_negative_pairs=int(config.calibration.num_negative_pairs),
            base_threshold_delta=float(config.calibration.base_threshold_delta),
            assign_threshold_floor_offset=float(config.calibration.assign_threshold_floor_offset),
            random_seed=int(config.debug.random_seed),
        )
    else:
        calibration = build_default_calibration(
            base_threshold_delta=float(config.calibration.base_threshold_delta),
            assign_threshold_floor_offset=float(config.calibration.assign_threshold_floor_offset),
        )

    discovery = discover_semantic_prototypes(
        embeddings=feature_bundle['embeddings'],
        view_stability=feature_bundle['view_stability'],
        calibration=calibration,
        config=config,
    )

    image_paths = feature_bundle['image_paths'].tolist()
    result_df = build_result_table(
        image_paths=image_paths,
        labels=discovery['labels'],
        prototype_ids=discovery['prototype_ids'],
        view_stability=feature_bundle['view_stability'],
        assignment_scores=discovery['assignment_scores'],
        assignment_margins=discovery['assignment_margins'],
        assignment_thresholds=discovery['assignment_thresholds'],
        prototype_confidences=discovery['prototype_confidences'],
        small_cluster_size=int(config.clustering.small_cluster_size),
    )
    cluster_summary = build_cluster_summary(result_df, int(discovery['num_prototypes_after_merge']))
    prototype_summary = build_prototype_summary(
        prototype_ids=discovery['prototype_ids'],
        prototype_thresholds=discovery['assignment_thresholds'],
        prototype_confidences=discovery['prototype_confidences'],
    )

    save_result_csv(paths['result_csv'], result_df)
    save_summary_json(paths['summary_json'], {
        'config': config.to_dict(),
        'num_samples': int(len(image_paths)),
        'model_name': str(config.encoder.model_name),
        'embedding_dim': int(feature_bundle['embeddings'].shape[1]),
        'num_views': int(feature_bundle['view_embeddings'].shape[1]),
        'calibration_enabled': bool(config.calibration.enabled),
        'calibration': calibration.to_dict(),
        'num_prototypes_before_merge': int(discovery['num_prototypes_before_merge']),
        'num_prototypes_after_merge': int(discovery['num_prototypes_after_merge']),
        'num_merge_events': int(discovery['num_merge_events']),
        **cluster_summary,
        **prototype_summary,
    })
    save_features_npz(paths['features_npz'], {
        'image_paths': feature_bundle['image_paths'],
        'sample_idx': feature_bundle['sample_idx'],
        'embeddings': feature_bundle['embeddings'],
        'view_embeddings': feature_bundle['view_embeddings'],
        'view_stability': feature_bundle['view_stability'],
        'labels': discovery['labels'],
        'prototype_ids': discovery['prototype_ids'],
        'assignment_scores': discovery['assignment_scores'],
        'assignment_margins': discovery['assignment_margins'],
        'assignment_thresholds': discovery['assignment_thresholds'],
        'prototype_confidences': discovery['prototype_confidences'],
        'center_scores': discovery['center_scores'],
        'support_scores': discovery['support_scores'],
        'ordered_indices': discovery['ordered_indices'],
        'created_new_mask': discovery['created_new_mask'],
        'labels_before_merge': discovery['labels_before_merge'],
        'prototype_ids_before_merge': discovery['prototype_ids_before_merge'],
        'assignment_thresholds_before_merge': discovery['assignment_thresholds_before_merge'],
    })
    save_prototype_npz(paths['prototype_npz'], {
        'prototype_ids': discovery['prototype_export_ids'],
        'prototype_centers': discovery['prototype_export_centers'],
        'prototype_thresholds': discovery['prototype_export_thresholds'],
        'prototype_confidences': discovery['prototype_export_confidences'],
        'prototype_sizes': discovery['prototype_export_sizes'],
        'prototype_support_indices': discovery['prototype_export_support_indices'],
        'prototype_member_counts': discovery['prototype_export_member_counts'],
    })

    return {
        'num_samples': int(len(image_paths)),
        'num_clusters': int(discovery['num_prototypes_after_merge']),
        'output_dir': str(config.output.output_dir),
        'features_path': paths['features_npz'],
        'result_csv_path': paths['result_csv'],
        'summary_json_path': paths['summary_json'],
        'prototype_npz_path': paths['prototype_npz'],
    }


def run_precluster_visual(config, stage: str = 'all'):
    _set_seed(int(config.debug.random_seed))
    ensure_dir(config.output.output_dir)
    selected_stage = str(stage).lower()
    if selected_stage == 'extract':
        return _run_extract_stage(config)
    if selected_stage == 'cluster':
        return _run_cluster_stage(config)
    if selected_stage == 'all':
        _run_extract_stage(config)
        return _run_cluster_stage(config)
    raise ValueError('unsupported stage: {}'.format(stage))
