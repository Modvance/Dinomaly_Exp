import random
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augment import build_view_transforms
from .cluster import build_cluster_summary, build_result_table, cluster_connected_components
from .data import MultiViewImagePathDataset, collate_multiview, discover_image_paths
from .encoder import extract_view_tokens, load_dinov2_model, resolve_layer_indices
from .features import aggregate_multiview, build_feature_bundle, compute_view_stability
from .graph import build_graph
from .io_utils import ensure_dir, save_features_npz, save_graph_npz, save_result_csv, save_summary_json


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_device(device_name: str) -> torch.device:
    if device_name.startswith('cuda') and not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device(device_name)


def _stack_feature_parts(feature_parts: List[Dict[str, torch.Tensor]], key: str) -> torch.Tensor:
    return torch.cat([part[key].detach().cpu() for part in feature_parts], dim=0)


def run_precluster_visual(config):
    _set_seed(int(config.debug.random_seed))
    ensure_dir(config.output.output_dir)
    device = _resolve_device(str(config.encoder.device))
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

    model = load_dinov2_model(config).to(device)
    layer_config = resolve_layer_indices(len(model.blocks), config.encoder.high_layer, config.encoder.mid_layers)
    autocast_enabled = bool(device.type == 'cuda' and str(config.encoder.precision).lower() in ['fp16', 'bf16'])

    all_object_parts = []
    all_structure_parts = []
    all_texture_parts = []
    all_embeddings = []
    all_view_stability = []

    for stacked_views, meta in dataloader:
        view_feature_parts = []
        for images in stacked_views:
            images = images.to(device)
            token_bundle = extract_view_tokens(model, images, layer_config, autocast_enabled=autocast_enabled)
            feature_bundle = build_feature_bundle(
                token_bundle['cls_tokens'],
                token_bundle['patch_tokens_high'],
                token_bundle['patch_tokens_mid'],
                config,
            )
            view_feature_parts.append(feature_bundle)

        object_views = [part['object_feature'] for part in view_feature_parts]
        structure_views = [part['structure_feature'] for part in view_feature_parts]
        texture_views = [part['texture_feature'] for part in view_feature_parts]
        embedding_views = [part['embedding'] for part in view_feature_parts]

        all_object_parts.append(aggregate_multiview(object_views, normalize=bool(config.features.normalize_each_feature)).detach().cpu())
        all_structure_parts.append(aggregate_multiview(structure_views, normalize=bool(config.features.normalize_each_feature)).detach().cpu())
        all_texture_parts.append(aggregate_multiview(texture_views, normalize=bool(config.features.normalize_each_feature)).detach().cpu())
        all_embeddings.append(aggregate_multiview(embedding_views, normalize=bool(config.features.normalize_final_feature)).detach().cpu())
        all_view_stability.append(compute_view_stability(embedding_views).detach().cpu())

    object_feature = torch.cat(all_object_parts, dim=0).numpy().astype(np.float32)
    structure_feature = torch.cat(all_structure_parts, dim=0).numpy().astype(np.float32)
    texture_feature = torch.cat(all_texture_parts, dim=0).numpy().astype(np.float32)
    embedding = torch.cat(all_embeddings, dim=0).numpy().astype(np.float32)
    view_stability = torch.cat(all_view_stability, dim=0).numpy().astype(np.float32)

    feature_bundle = {
        'object_feature': object_feature,
        'structure_feature': structure_feature,
        'texture_feature': texture_feature,
        'embedding': embedding,
    }
    graph_result = build_graph(feature_bundle, config)
    num_components, labels = cluster_connected_components(len(image_paths), graph_result['rows'], graph_result['cols'], graph_result['edge_scores'])
    result_df = build_result_table(image_paths, labels, view_stability, graph_result['rows'], graph_result['cols'], graph_result['edge_scores'], int(config.clustering.small_cluster_size))
    cluster_summary = build_cluster_summary(result_df, num_components)

    result_csv_path = '{}/{}'.format(config.output.output_dir, config.output.result_csv)
    summary_json_path = '{}/{}'.format(config.output.output_dir, config.output.summary_json)
    features_npz_path = '{}/{}'.format(config.output.output_dir, config.output.features_npz)
    graph_npz_path = '{}/{}'.format(config.output.output_dir, config.output.graph_npz)

    save_result_csv(result_csv_path, result_df)
    save_summary_json(summary_json_path, {
        'config': config.to_dict(),
        'num_samples': int(len(image_paths)),
        'model_name': str(config.encoder.model_name),
        'layer_config': layer_config,
        'feature_dims': {
            'object': int(object_feature.shape[1]),
            'structure': int(structure_feature.shape[1]),
            'texture': int(texture_feature.shape[1]),
            'embedding': int(embedding.shape[1]),
        },
        'selected_k': int(graph_result['k']),
        'requested_k': int(graph_result['requested_k']),
        'candidate_edge_count': int(graph_result['candidate_edge_count']),
        'retained_edge_count': int(graph_result['retained_edge_count']),
        **cluster_summary,
    })
    save_features_npz(features_npz_path, {
        'embeddings': embedding,
        'object_features': object_feature,
        'structure_features': structure_feature,
        'texture_features': texture_feature,
        'view_stability': view_stability,
        'labels': np.asarray(labels, dtype=np.int64),
    })
    save_graph_npz(graph_npz_path, graph_result['rows'], graph_result['cols'], graph_result['edge_scores'], extras=graph_result['extras'])

    return {
        'num_samples': int(len(image_paths)),
        'num_clusters': int(num_components),
        'output_dir': str(config.output.output_dir),
    }
