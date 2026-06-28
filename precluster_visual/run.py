import random
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augment import build_view_transforms
from .calibrator import fit_similarity_calibrator
from .cluster import build_cluster_summary, build_result_table, cluster_connected_components
from .data import MultiViewImagePathDataset, collate_multiview, discover_image_paths
from .encoder import SiglipImageEncoder
from .features import aggregate_multiview, compute_view_stability, stack_view_embeddings
from .graph import build_graph
from .io_utils import ensure_dir, save_features_npz, save_graph_npz, save_result_csv, save_summary_json


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
        'graph_npz': '{}/{}'.format(config.output.output_dir, config.output.graph_npz),
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
    calibrator = fit_similarity_calibrator(
        view_embeddings=feature_bundle['view_embeddings'],
        embeddings=feature_bundle['embeddings'],
        num_bins=int(config.calibration.num_bins),
        num_negative_pairs=int(config.calibration.num_negative_pairs),
        random_seed=int(config.debug.random_seed),
    ) if bool(config.calibration.enabled) else None

    graph_result = build_graph(feature_bundle, config, calibrator=calibrator)
    image_paths = feature_bundle['image_paths'].tolist()
    num_components, labels = cluster_connected_components(len(image_paths), graph_result['rows'], graph_result['cols'], graph_result['edge_scores'])
    result_df = build_result_table(
        image_paths,
        labels,
        feature_bundle['view_stability'],
        graph_result['rows'],
        graph_result['cols'],
        graph_result['edge_scores'],
        int(config.clustering.small_cluster_size),
    )
    cluster_summary = build_cluster_summary(result_df, num_components)

    save_result_csv(paths['result_csv'], result_df)
    save_summary_json(paths['summary_json'], {
        'config': config.to_dict(),
        'num_samples': int(len(image_paths)),
        'model_name': str(config.encoder.model_name),
        'embedding_dim': int(feature_bundle['embeddings'].shape[1]),
        'num_views': int(feature_bundle['view_embeddings'].shape[1]),
        'selected_k': int(graph_result['k']),
        'requested_k': int(graph_result['requested_k']),
        'candidate_edge_count': int(graph_result['candidate_edge_count']),
        'retained_edge_count': int(graph_result['retained_edge_count']),
        'calibration_enabled': bool(config.calibration.enabled),
        'calibration_fitted': bool(calibrator is not None),
        **cluster_summary,
    })
    save_features_npz(paths['features_npz'], {
        'image_paths': feature_bundle['image_paths'],
        'sample_idx': feature_bundle['sample_idx'],
        'embeddings': feature_bundle['embeddings'],
        'view_embeddings': feature_bundle['view_embeddings'],
        'view_stability': feature_bundle['view_stability'],
        'labels': np.asarray(labels, dtype=np.int64),
    })
    save_graph_npz(paths['graph_npz'], graph_result['rows'], graph_result['cols'], graph_result['edge_scores'], extras=graph_result['extras'])

    return {
        'num_samples': int(len(image_paths)),
        'num_clusters': int(num_components),
        'output_dir': str(config.output.output_dir),
        'features_path': paths['features_npz'],
        'result_csv_path': paths['result_csv'],
        'summary_json_path': paths['summary_json'],
        'graph_npz_path': paths['graph_npz'],
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
