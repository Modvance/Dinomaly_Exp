"""Paired, no-training TailGuard memory evaluation from one Full checkpoint.

The evaluator shares the retained training set and every model forward between
the Full and raw-tail registries.  Only pseudo-class routing/memory membership
differs, which makes the resulting comparison a direct audit of P.
"""

import argparse
import datetime as dt
import glob
import hashlib
import json
import math
import os
import platform
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dinomaly_train_base import (
    _checkpoint_args_get,
    build_datasets,
    load_model_from_train_checkpoint,
    load_train_checkpoint_metadata,
    rebuild_pruned_train_loaders,
)
from one_shot_memory import compute_memory_patch_scores, pool_patch_scores, resize_memory_map
from tailguard_artifacts import save_tailguard_memory_artifacts
from tailguard_dataset_profiles import (
    MVTec_PROFILE,
    dataset_profile_names,
    get_dataset_profile,
    profile_provenance,
    validate_profile_contract,
)
from tailguard_memory import route_evidence_mask
from utils import cal_anomaly_maps, compute_image_level_scores, compute_pro, f1_score_max, get_gaussian_kernel


PROTOCOL = {
    'memory_topk_ratio': 0.10,
    'memory_fusion_lambda': 0.25,
    'route_margin_threshold': 0.04255223274230957,
    'memory_min_class_members': 2,
}
VARIANTS = ('full', 'h_raw_e')
METRIC_KEYS = ('I-AUROC', 'I-AP', 'I-F1', 'P-AUROC', 'P-AP', 'P-F1', 'P-AUPRO')


def _utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    # bool is an int subclass; preserve JSON true/false before integer handling.
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(_json_safe(payload), file, indent=2, ensure_ascii=False)


def _sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        while True:
            block = file.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sha256_rows(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(('{}:{}\n'.format(int(row[0]), int(row[1]))).encode('ascii'))
    return digest.hexdigest()


def _integer(value, name, minimum):
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < minimum:
        raise ValueError('{} must be an integer >= {}: {}'.format(name, minimum, value))
    return int(numeric)


def _fixed_float(value, name, expected):
    numeric = float(value)
    if not math.isfinite(numeric) or not math.isclose(numeric, expected, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError('{} is frozen at {}: {}'.format(name, expected, value))
    return numeric


def _fixed_integer(value, name, expected):
    numeric = _integer(value, name, 1)
    if numeric != expected:
        raise ValueError('{} is frozen at {}: {}'.format(name, expected, value))
    return numeric


def _resolve_profile(parsed_args, checkpoint_metadata):
    checkpoint_profile_name = _checkpoint_args_get(checkpoint_metadata.get('args'), 'dataset_profile', None)
    if parsed_args.dataset_profile is not None:
        profile = get_dataset_profile(parsed_args.dataset_profile)
    elif checkpoint_profile_name is not None:
        profile = get_dataset_profile(checkpoint_profile_name)
    else:
        profile = MVTec_PROFILE
    validate_profile_contract(
        profile,
        checkpoint_profile_name,
        _checkpoint_args_get(checkpoint_metadata.get('args'), 'dataset_item_list', None),
        'checkpoint',
    )
    return profile


def _load_registry(pseudoclass_dir, profile, variant):
    directory = os.path.realpath(pseudoclass_dir)
    members_path = os.path.join(directory, 'pseudo_class_members.csv')
    classes_path = os.path.join(directory, 'pseudo_classes.csv')
    registry_path = os.path.join(directory, 'pseudo_class_registry.json')
    for path in (members_path, classes_path, registry_path):
        if not os.path.isfile(path):
            raise FileNotFoundError('{} registry artifact does not exist: {}'.format(variant, path))

    members = pd.read_csv(members_path)
    classes = pd.read_csv(classes_path)
    required_members = {'sample_idx', 'class_id', 'base_idx', 'pseudo_class_id', 'pseudo_class_type'}
    required_classes = {'pseudo_class_id', 'pseudo_class_type', 'num_members'}
    missing_members = required_members.difference(members.columns)
    missing_classes = required_classes.difference(classes.columns)
    if missing_members or missing_classes:
        raise ValueError(
            '{} registry columns are incomplete (members={}, classes={})'.format(
                variant, sorted(missing_members), sorted(missing_classes)
            )
        )
    for column in ('sample_idx', 'class_id', 'base_idx', 'pseudo_class_id'):
        members[column] = members[column].astype(int)
    classes['pseudo_class_id'] = classes['pseudo_class_id'].astype(int)
    classes['num_members'] = classes['num_members'].astype(int)
    if members.duplicated(['class_id', 'base_idx']).any() or members['sample_idx'].duplicated().any():
        raise ValueError('{} registry contains duplicate stable identities'.format(variant))
    if classes['pseudo_class_id'].duplicated().any() or (classes['num_members'] <= 0).any():
        raise ValueError('{} registry contains invalid pseudo-classes'.format(variant))
    member_counts = members.groupby('pseudo_class_id').size().to_dict()
    class_counts = classes.set_index('pseudo_class_id')['num_members'].to_dict()
    if member_counts != class_counts:
        raise ValueError('{} registry membership counts do not match pseudo_classes.csv'.format(variant))
    member_types = members.groupby('pseudo_class_id')['pseudo_class_type'].nunique().to_dict()
    if any(value != 1 for value in member_types.values()):
        raise ValueError('{} registry mixes types within a pseudo-class'.format(variant))

    with open(registry_path, 'r', encoding='utf-8') as file:
        registry = json.load(file)
    provenance = registry.get('dataset_provenance')
    if provenance is not None:
        if not isinstance(provenance, dict):
            raise ValueError('{} registry dataset_provenance must be an object'.format(variant))
        validate_profile_contract(
            profile,
            provenance.get('profile_name'),
            provenance.get('item_list'),
            '{} registry'.format(variant),
        )
    return {
        'variant': variant,
        'dir': directory,
        'members': members.sort_values(['class_id', 'base_idx'], kind='mergesort').reset_index(drop=True),
        'classes': classes.sort_values('pseudo_class_id', kind='mergesort').reset_index(drop=True),
        'registry': registry,
        'paths': {
            'members': members_path,
            'classes': classes_path,
            'registry': registry_path,
        },
    }


def validate_shared_registries(registries, num_classes=None):
    """Require identical retained identities and canonical sample indices."""
    reference = registries['full']['members']
    reference_map = {
        (int(row.class_id), int(row.base_idx)): int(row.sample_idx)
        for row in reference.itertuples(index=False)
    }
    for variant in VARIANTS[1:]:
        current = registries[variant]['members']
        current_map = {
            (int(row.class_id), int(row.base_idx)): int(row.sample_idx)
            for row in current.itertuples(index=False)
        }
        if current_map != reference_map:
            missing = sorted(set(reference_map).difference(current_map))[:10]
            unexpected = sorted(set(current_map).difference(reference_map))[:10]
            remapped = sorted(
                key for key in set(reference_map).intersection(current_map)
                if reference_map[key] != current_map[key]
            )[:10]
            raise ValueError(
                '{} registry does not share the Full retained set '
                '(missing={}, unexpected={}, remapped={})'.format(
                    variant, missing, unexpected, remapped
                )
            )
    stable_rows = sorted(reference_map)
    retained_index_map = {
        int(class_id): sorted(
            int(base_idx) for (identity_class, base_idx) in stable_rows
            if identity_class == class_id
        )
        for class_id in sorted(set(key[0] for key in stable_rows))
    }
    if num_classes is not None:
        class_count = _integer(num_classes, 'num_classes', 1)
        unexpected_classes = sorted(set(retained_index_map).difference(range(class_count)))
        if unexpected_classes:
            raise ValueError('paired registries contain unknown class ids: {}'.format(unexpected_classes))
        # Missing classes mean purification retained zero samples, not all samples.
        retained_index_map = {
            class_id: retained_index_map.get(class_id, [])
            for class_id in range(class_count)
        }
    return {
        'num_retained_samples': len(stable_rows),
        'stable_identity_sha256': _sha256_rows(stable_rows),
        'sample_idx_mapping_sha256': hashlib.sha256(
            ''.join(
                '{}:{}:{}\n'.format(key[0], key[1], reference_map[key])
                for key in stable_rows
            ).encode('ascii')
        ).hexdigest(),
        'retained_index_map': retained_index_map,
    }


def _normalize_rows(tensor):
    return F.normalize(tensor.float(), dim=-1)


def _encoder_features_once(model, images):
    """Extract Dinomaly patch features and final DINOv2 CLS in one encoder pass."""
    encoder = getattr(model, 'encoder', None)
    if encoder is None or not hasattr(encoder, 'prepare_tokens'):
        raise ValueError('paired evaluation requires a ViTill model with a DINOv2 encoder')
    tokens = encoder.prepare_tokens(images)
    captured = []
    target_layers = set(int(value) for value in model.target_layers)
    for index, block in enumerate(encoder.blocks):
        tokens = block(tokens)
        if index in target_layers:
            captured.append(tokens)
    if len(captured) != len(model.target_layers):
        raise ValueError('model target layers were not all observed during the encoder pass')
    cls_embeddings = _normalize_rows(encoder.norm(tokens)[:, 0])
    side = int(math.sqrt(captured[0].shape[1] - 1 - encoder.num_register_tokens))
    reconstruction_tokens = captured
    if model.remove_class_token:
        reconstruction_tokens = [
            value[:, 1 + encoder.num_register_tokens:, :] for value in captured
        ]
    patch_tokens = model.fuse_feature(reconstruction_tokens)
    if not model.remove_class_token:
        patch_tokens = patch_tokens[:, 1 + encoder.num_register_tokens:, :]
    return (
        _normalize_rows(patch_tokens),
        (side, side),
        cls_embeddings,
        reconstruction_tokens,
    )


def _joint_test_forward_once(model, images, gaussian_kernel, resize_mask):
    """Produce reconstruction, patch and CLS features with one model traversal."""
    patch_tokens, spatial_size, cls_embeddings, encoder_tokens = _encoder_features_once(model, images)
    decoder_tokens = model.fuse_feature(encoder_tokens)
    for block in model.bottleneck:
        decoder_tokens = block(decoder_tokens)
    attention_mask = (
        model.generate_mask(spatial_size[0], decoder_tokens.device)
        if model.mask_neighbor_size > 0 else None
    )
    decoder_features = []
    for block in model.decoder:
        decoder_tokens = block(decoder_tokens, attn_mask=attention_mask)
        decoder_features.append(decoder_tokens)
    decoder_features = decoder_features[::-1]
    encoder_outputs = [
        model.fuse_feature([encoder_tokens[index] for index in indices])
        for indices in model.fuse_layer_encoder
    ]
    decoder_outputs = [
        model.fuse_feature([decoder_features[index] for index in indices])
        for indices in model.fuse_layer_decoder
    ]
    register_tokens = model.encoder.num_register_tokens
    if not model.remove_class_token:
        encoder_outputs = [value[:, 1 + register_tokens:, :] for value in encoder_outputs]
        decoder_outputs = [value[:, 1 + register_tokens:, :] for value in decoder_outputs]
    encoder_maps = [
        value.permute(0, 2, 1).reshape(images.shape[0], -1, *spatial_size).contiguous()
        for value in encoder_outputs
    ]
    decoder_maps = [
        value.permute(0, 2, 1).reshape(images.shape[0], -1, *spatial_size).contiguous()
        for value in decoder_outputs
    ]
    reconstruction_map, _ = cal_anomaly_maps(encoder_maps, decoder_maps, images.shape[-1])
    reconstruction_map = F.interpolate(
        reconstruction_map, size=resize_mask, mode='bilinear', align_corners=False
    )
    reconstruction_map = gaussian_kernel(reconstruction_map)
    return reconstruction_map, patch_tokens, spatial_size, cls_embeddings


@torch.no_grad()
def validate_joint_forward_equivalence(model, images, gaussian_kernel, resize_mask):
    """Fail before evaluation if the shared forward changes reconstruction."""
    from one_shot_memory import extract_encoder_patch_tokens
    from utils import infer_anomaly_map_batch

    reference_map, _ = infer_anomaly_map_batch(
        model, images, gaussian_kernel, resize_mask=resize_mask
    )
    reference_patches, reference_size = extract_encoder_patch_tokens(model, images)
    joint_map, joint_patches, joint_size, _ = _joint_test_forward_once(
        model, images, gaussian_kernel, resize_mask
    )
    map_error = float((reference_map - joint_map).abs().max().item())
    patch_error = float((reference_patches - joint_patches).abs().max().item())
    if tuple(reference_size) != tuple(joint_size) or map_error > 1e-5 or patch_error > 1e-5:
        raise RuntimeError(
            'shared-forward equivalence failed '
            '(map_error={}, patch_error={}, reference_size={}, joint_size={})'.format(
                map_error, patch_error, reference_size, joint_size
            )
        )
    return {
        'max_abs_reconstruction_map_error': map_error,
        'max_abs_patch_feature_error': patch_error,
        'spatial_size': list(joint_size),
    }


def _allocate_patch_quotas(capacities, max_patches):
    total = int(sum(capacities))
    target = total if max_patches <= 0 else min(total, int(max_patches))
    quotas = [0] * len(capacities)
    active = [index for index, capacity in enumerate(capacities) if capacity > 0]
    remaining = target
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        next_active = []
        for position, index in enumerate(active):
            added = min(share, capacities[index] - quotas[index], remaining)
            quotas[index] += added
            remaining -= added
            if quotas[index] < capacities[index]:
                next_active.append(index)
            if remaining == 0:
                next_active.extend(active[position + 1:])
                break
        active = next_active
    return quotas


def _sample_patches(features, quota):
    if quota <= 0:
        return features[:0]
    if quota >= features.shape[0]:
        return features
    indices = torch.linspace(0, features.shape[0] - 1, steps=quota).round().long()
    return features.index_select(0, indices)


@torch.no_grad()
def extract_shared_train_cache(model, train_loader, shared, registries, device):
    expected = {
        (int(row.class_id), int(row.base_idx))
        for row in registries['full']['members'].itertuples(index=False)
    }
    tail_required = set()
    for variant in VARIANTS:
        members = registries[variant]['members']
        tail_required.update(
            (int(row.class_id), int(row.base_idx))
            for row in members.loc[members['pseudo_class_type'] == 'tail'].itertuples(index=False)
        )
    cache = {}
    loader_keys = set()
    was_training = model.training
    model.eval()
    try:
        for images, _, meta in train_loader:
            images = images.to(device)
            patch_features, spatial_size, cls_embeddings, _ = _encoder_features_once(model, images)
            for index in range(images.shape[0]):
                key = (int(meta['class_id'][index]), int(meta['base_idx'][index]))
                if key in loader_keys:
                    raise ValueError('shared train loader yielded duplicate identity {}'.format(key))
                loader_keys.add(key)
                if key not in expected:
                    raise ValueError('shared train loader yielded unexpected identity {}'.format(key))
                record = {'cls': cls_embeddings[index].detach().cpu().contiguous()}
                if key in tail_required:
                    record.update({
                        'patches': patch_features[index].detach().cpu().contiguous(),
                        'spatial_size': tuple(int(value) for value in spatial_size),
                    })
                cache[key] = record
    finally:
        if was_training:
            model.train()
    if loader_keys != expected:
        raise ValueError(
            'shared train loader identities differ from paired registries '
            '(missing={}, unexpected={})'.format(
                sorted(expected.difference(loader_keys))[:10],
                sorted(loader_keys.difference(expected))[:10],
            )
        )
    if len(cache) != shared['num_retained_samples']:
        raise ValueError('shared train cache size does not match retained-set audit')
    return cache


def build_memory_system_from_cache(registry, train_cache, max_patches):
    members = registry['members']
    classes = registry['classes']
    entries = {}
    for class_row in classes.itertuples(index=False):
        pseudo_class_id = int(class_row.pseudo_class_id)
        pseudo_class_type = str(class_row.pseudo_class_type)
        class_members = members.loc[members['pseudo_class_id'] == pseudo_class_id].sort_values(
            'sample_idx', kind='mergesort'
        )
        keys = [
            (int(row.class_id), int(row.base_idx))
            for row in class_members.itertuples(index=False)
        ]
        records = [train_cache[key] for key in keys]
        cls_matrix = torch.stack([record['cls'] for record in records], dim=0)
        entry = {
            'pseudo_class_id': pseudo_class_id,
            'pseudo_class_type': pseudo_class_type,
            'num_members': len(records),
            'member_sample_idx': class_members['sample_idx'].astype(int).tolist(),
            'prototype_cls': _normalize_rows(cls_matrix.mean(dim=0, keepdim=True))[0].cpu().contiguous(),
            'member_cls': cls_matrix.cpu().contiguous(),
            'has_memory_bank': pseudo_class_type == 'tail',
            'num_patches': 0,
            'feature_dim': 0,
            'spatial_size': None,
            'member_patch_quotas': [],
        }
        if pseudo_class_type == 'tail':
            dimensions = {int(record['patches'].shape[1]) for record in records}
            spatial_sizes = {record['spatial_size'] for record in records}
            if len(dimensions) != 1 or len(spatial_sizes) != 1:
                raise ValueError('{} tail class {} has inconsistent patch shapes'.format(
                    registry['variant'], pseudo_class_id
                ))
            capacities = [int(record['patches'].shape[0]) for record in records]
            quotas = _allocate_patch_quotas(capacities, max_patches)
            selected = [
                _sample_patches(record['patches'], quota)
                for record, quota in zip(records, quotas)
                if quota > 0
            ]
            if not selected:
                raise ValueError('{} tail class {} has no memory patches'.format(
                    registry['variant'], pseudo_class_id
                ))
            features = torch.cat(selected, dim=0).contiguous().cpu()
            entry.update({
                'features': features,
                'num_patches': int(features.shape[0]),
                'feature_dim': int(features.shape[1]),
                'spatial_size': next(iter(spatial_sizes)),
                'member_patch_quotas': [
                    {
                        'sample_idx': int(sample_idx),
                        'num_available_patches': int(capacity),
                        'num_selected_patches': int(quota),
                    }
                    for sample_idx, capacity, quota in zip(
                        class_members['sample_idx'].astype(int), capacities, quotas
                    )
                ],
            })
        elif pseudo_class_type != 'head':
            raise ValueError('unknown pseudo-class type: {}'.format(pseudo_class_type))
        entries[pseudo_class_id] = entry
    return {
        'schema_version': 1,
        'class_entries': entries,
        'num_pseudo_classes': len(entries),
        'num_head_pseudo_classes': sum(row['pseudo_class_type'] == 'head' for row in entries.values()),
        'num_tail_pseudo_classes': sum(row['pseudo_class_type'] == 'tail' for row in entries.values()),
        'num_tail_memory_banks': sum(row['has_memory_bank'] for row in entries.values()),
        'max_patches_per_class': int(max_patches),
    }


def _routing_cache(memory_system, device):
    entries = memory_system['class_entries']
    class_ids = sorted(int(value) for value in entries)
    prototypes = torch.stack([entries[value]['prototype_cls'] for value in class_ids]).to(device)
    return {
        'class_ids': class_ids,
        'prototypes': _normalize_rows(prototypes),
        'class_types': [str(entries[value]['pseudo_class_type']) for value in class_ids],
    }


def score_cached_features(patch_features, spatial_size, cls_embeddings, memory_system, args,
                          device, routing_cache, device_memory_cache):
    similarities = cls_embeddings @ routing_cache['prototypes'].T
    best_positions = similarities.argmax(dim=1)
    best_similarity = similarities.gather(1, best_positions[:, None])[:, 0]
    if similarities.shape[1] > 1:
        second_similarity = torch.topk(similarities, k=2, dim=1).values[:, 1]
    else:
        second_similarity = torch.full_like(best_similarity, float('nan'))
    scores, maps, eligible, predicted_ids, predicted_types = [], [], [], [], []
    for index, position in enumerate(best_positions.tolist()):
        pseudo_class_id = int(routing_cache['class_ids'][position])
        pseudo_class_type = routing_cache['class_types'][position]
        entry = memory_system['class_entries'][pseudo_class_id]
        predicted_ids.append(pseudo_class_id)
        predicted_types.append(pseudo_class_type)
        supported_tail = (
            pseudo_class_type == 'tail'
            and int(entry['num_members']) >= int(args.tg_memory_min_class_members)
        )
        if not supported_tail:
            if pseudo_class_type == 'tail':
                predicted_types[-1] = 'tail_unsupported'
            scores.append(torch.tensor(0.0, device=device))
            maps.append(torch.zeros(spatial_size, device=device))
            eligible.append(0)
            continue
        memory_features = device_memory_cache.setdefault(
            pseudo_class_id, entry['features'].to(device)
        )
        patch_scores = compute_memory_patch_scores(
            patch_features[index], memory_features, chunk_size=int(args.tg_mem_chunk_size)
        )
        scores.append(pool_patch_scores(patch_scores, topk_ratio=args.tg_memory_topk_ratio))
        maps.append(patch_scores.reshape(spatial_size))
        eligible.append(1)
    return {
        'memory_scores': torch.stack(scores),
        'memory_patch_maps': torch.stack(maps),
        'route_eligible': torch.tensor(eligible, dtype=torch.int64, device=device),
        'predicted_pseudo_class_id': torch.tensor(predicted_ids, dtype=torch.int64),
        'predicted_pseudo_class_type': predicted_types,
        'top1_similarity': best_similarity.detach().cpu(),
        'second_similarity': second_similarity.detach().cpu(),
        'similarity_margin': (best_similarity - second_similarity).detach().cpu(),
    }


def _metrics_for_class(class_id, class_name, predictions, num_samples):
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows = {}
    for mode, values in predictions.items():
        gt_sp = torch.cat(values['gt_sp']).flatten().numpy()
        pred_sp = torch.cat(values['pred_sp']).flatten().numpy()
        gt_px = torch.cat(values['gt_px']).numpy()
        pred_px = torch.cat(values['pred_px']).numpy()
        rows[mode] = {
            'class_id': int(class_id),
            'class_name': str(class_name),
            'I-AUROC': float(roc_auc_score(gt_sp, pred_sp)),
            'I-AP': float(average_precision_score(gt_sp, pred_sp)),
            'I-F1': float(f1_score_max(gt_sp, pred_sp)),
            'P-AUROC': float(roc_auc_score(gt_px.ravel(), pred_px.ravel())),
            'P-AP': float(average_precision_score(gt_px.ravel(), pred_px.ravel())),
            'P-F1': float(f1_score_max(gt_px.ravel(), pred_px.ravel())),
            'P-AUPRO': float(compute_pro(gt_px, pred_px)),
            'num_samples': int(num_samples),
        }
    return rows


@torch.no_grad()
def run_paired_test_evaluation(model, test_data_list, item_list, memory_systems, args, device):
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    score_rows = {variant: [] for variant in VARIANTS}
    per_class = {variant: {mode: [] for mode in ('recon', 'fused')} for variant in VARIANTS}
    route_caches = {
        variant: _routing_cache(memory_systems[variant], device)
        for variant in VARIANTS
    }
    device_memory_caches = {variant: {} for variant in VARIANTS}
    was_training = model.training
    model.eval()
    equivalence_audit = None
    evaluation_started = time.time()
    try:
        for class_id, (class_name, test_data) in enumerate(zip(item_list, test_data_list)):
            class_started = time.time()
            loader = torch.utils.data.DataLoader(
                test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
            )
            predictions = {
                variant: {
                    mode: {'gt_sp': [], 'pred_sp': [], 'gt_px': [], 'pred_px': []}
                    for mode in ('recon', 'fused')
                }
                for variant in VARIANTS
            }
            sample_count = 0
            for images, gt, label, img_paths in loader:
                images = images.to(device)
                gt = gt.to(device)
                if equivalence_audit is None:
                    equivalence_audit = validate_joint_forward_equivalence(
                        model, images[:1], gaussian_kernel, args.diag_resize_mask
                    )
                reconstruction_map, patch_features, spatial_size, cls_embeddings = _joint_test_forward_once(
                    model, images, gaussian_kernel, args.diag_resize_mask
                )
                gt = F.interpolate(gt, size=args.diag_resize_mask, mode='nearest').bool()
                if gt.shape[1] > 1:
                    gt = torch.max(gt, dim=1, keepdim=True)[0]
                reconstruction_scores = compute_image_level_scores(
                    reconstruction_map, max_ratio=args.diag_max_ratio
                )
                for variant in VARIANTS:
                    memory = score_cached_features(
                        patch_features,
                        spatial_size,
                        cls_embeddings,
                        memory_systems[variant],
                        args,
                        device,
                        route_caches[variant],
                        device_memory_caches[variant],
                    )
                    memory_map = resize_memory_map(
                        memory['memory_patch_maps'], reconstruction_map.shape[-2:]
                    ).unsqueeze(1)
                    margin = memory['similarity_margin'].to(device)
                    applied = route_evidence_mask(
                        memory['route_eligible'], margin, args.tg_memory_route_margin_threshold
                    )
                    fused_scores = torch.where(
                        applied,
                        reconstruction_scores + args.tg_memory_fusion_lambda * memory['memory_scores'],
                        reconstruction_scores,
                    )
                    fused_maps = torch.where(
                        applied[:, None, None, None],
                        reconstruction_map + args.tg_memory_fusion_lambda * memory_map,
                        reconstruction_map,
                    )
                    modes = {
                        'recon': (reconstruction_scores, reconstruction_map),
                        'fused': (fused_scores, fused_maps),
                    }
                    for mode, (scores, maps) in modes.items():
                        predictions[variant][mode]['gt_sp'].append(label.cpu())
                        predictions[variant][mode]['pred_sp'].append(scores.detach().cpu())
                        predictions[variant][mode]['gt_px'].append(gt[:, 0].detach().cpu())
                        predictions[variant][mode]['pred_px'].append(maps[:, 0].detach().cpu())
                    for index in range(images.shape[0]):
                        score_rows[variant].append({
                            'class_id': int(class_id),
                            'class_name': str(class_name),
                            'img_path': str(img_paths[index]),
                            'label': int(label[index].item()),
                            'recon_score': float(reconstruction_scores[index].item()),
                            'memory_score': float(memory['memory_scores'][index].item()),
                            'final_score': float(fused_scores[index].item()),
                            'route_eligible': int(memory['route_eligible'][index].item()),
                            'mem_applied': int(applied[index].item()),
                            'predicted_pseudo_class_id': int(
                                memory['predicted_pseudo_class_id'][index].item()
                            ),
                            'predicted_pseudo_class_type': memory['predicted_pseudo_class_type'][index],
                            'top1_similarity': float(memory['top1_similarity'][index].item()),
                            'second_similarity': float(memory['second_similarity'][index].item()),
                            'similarity_margin': float(memory['similarity_margin'][index].item()),
                        })
                sample_count += int(images.shape[0])
            reference_variant = VARIANTS[0]
            reference_metrics = _metrics_for_class(
                class_id, class_name, predictions[reference_variant], sample_count
            )
            for mode in reference_metrics:
                per_class[reference_variant][mode].append(reference_metrics[mode])
            for variant in VARIANTS[1:]:
                fused_metrics = _metrics_for_class(
                    class_id,
                    class_name,
                    {'fused': predictions[variant]['fused']},
                    sample_count,
                )
                per_class[variant]['recon'].append(dict(reference_metrics['recon']))
                per_class[variant]['fused'].append(fused_metrics['fused'])
            print(
                'paired evaluation class [{}/{}] {}: {} samples, {:.1f}s'.format(
                    class_id + 1,
                    len(item_list),
                    class_name,
                    sample_count,
                    time.time() - class_started,
                ),
                flush=True,
            )
    finally:
        if was_training:
            model.train()

    results = {}
    for variant in VARIANTS:
        metrics_by_mode = {}
        for mode in ('recon', 'fused'):
            frame = pd.DataFrame(per_class[variant][mode])
            metrics_by_mode[mode] = {
                'summary': {key: float(frame[key].mean()) for key in METRIC_KEYS},
                'per_class_metrics': per_class[variant][mode],
            }
        score_frame = pd.DataFrame(score_rows[variant])
        summary = dict(metrics_by_mode['fused']['summary'])
        summary.update({
            'num_pseudo_classes': int(memory_systems[variant]['num_pseudo_classes']),
            'num_head_pseudo_classes': int(memory_systems[variant]['num_head_pseudo_classes']),
            'num_tail_pseudo_classes': int(memory_systems[variant]['num_tail_pseudo_classes']),
            'num_tail_memory_banks': int(memory_systems[variant]['num_tail_memory_banks']),
            'route_eligible_ratio': float(score_frame['route_eligible'].mean()),
            'memory_applied_ratio': float(score_frame['mem_applied'].mean()),
            'route_margin_threshold': args.tg_memory_route_margin_threshold,
            'memory_min_class_members': args.tg_memory_min_class_members,
        })
        results[variant] = {
            'score_df': score_frame,
            'per_class_metrics': per_class[variant]['fused'],
            'summary': summary,
            'metrics_by_mode': metrics_by_mode,
        }
    print(
        'paired test evaluation finished in {:.1f}s'.format(time.time() - evaluation_started),
        flush=True,
    )
    return results, equivalence_audit


def build_paired_score_audit(results):
    key_columns = ['class_id', 'class_name', 'img_path', 'label']
    frames = []
    for variant in VARIANTS:
        frame = results[variant]['score_df'].copy()
        rename = {
            column: '{}_{}'.format(variant, column)
            for column in frame.columns if column not in key_columns
        }
        frames.append(frame.rename(columns=rename))
    paired = frames[0].merge(frames[1], on=key_columns, how='outer', validate='one_to_one', indicator=True)
    if not (paired['_merge'] == 'both').all():
        raise ValueError('Full and raw-tail test score rows do not align one-to-one')
    paired = paired.drop(columns=['_merge'])
    for field in ('recon_score', 'memory_score', 'final_score', 'route_eligible', 'mem_applied'):
        paired['full_minus_h_raw_e_{}'.format(field)] = (
            paired['full_{}'.format(field)] - paired['h_raw_e_{}'.format(field)]
        )
    return paired


def _registry_audit_rows(registries, shared):
    rows = []
    for variant in VARIANTS:
        registry = registries[variant]
        members = registry['members']
        classes = registry['classes']
        rows.append({
            'variant': variant,
            'pseudoclass_dir': registry['dir'],
            'num_retained_samples': len(members),
            'num_pseudo_classes': len(classes),
            'num_head_pseudo_classes': int((classes['pseudo_class_type'] == 'head').sum()),
            'num_tail_pseudo_classes': int((classes['pseudo_class_type'] == 'tail').sum()),
            'num_tail_members': int((members['pseudo_class_type'] == 'tail').sum()),
            'num_supported_tail_classes': int((
                (classes['pseudo_class_type'] == 'tail')
                & (classes['num_members'] >= PROTOCOL['memory_min_class_members'])
            ).sum()),
            'stable_identity_sha256': shared['stable_identity_sha256'],
            'members_csv_sha256': _sha256_file(registry['paths']['members']),
            'classes_csv_sha256': _sha256_file(registry['paths']['classes']),
            'registry_json_sha256': _sha256_file(registry['paths']['registry']),
        })
    return rows


def _metric_audit_rows(results):
    rows = []
    for mode in ('recon', 'fused'):
        for metric in METRIC_KEYS:
            full_value = results['full']['metrics_by_mode'][mode]['summary'][metric]
            raw_value = results['h_raw_e']['metrics_by_mode'][mode]['summary'][metric]
            rows.append({
                'mode': mode,
                'metric': metric,
                'full': full_value,
                'h_raw_e': raw_value,
                'full_minus_h_raw_e': full_value - raw_value,
            })
    return rows


def _environment_provenance(device):
    gpu = None
    if device.type == 'cuda':
        gpu = {
            'index': int(device.index or 0),
            'name': torch.cuda.get_device_name(device),
            'cuda_runtime': torch.version.cuda,
        }
    return {
        'python': sys.version,
        'platform': platform.platform(),
        'torch': torch.__version__,
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'device': str(device),
        'gpu': gpu,
        'argv': list(sys.argv),
    }


def _reject_existing_output(output_dir):
    if os.path.exists(output_dir):
        raise FileExistsError('output_dir must not already exist: {}'.format(output_dir))
    in_progress = sorted(glob.glob(output_dir + '.inprogress-*'))
    if in_progress:
        raise FileExistsError('paired evaluation has existing in-progress output: {}'.format(in_progress[0]))


def paired_evaluate(parsed_args):
    started_at = _utc_now()
    checkpoint_path = os.path.realpath(parsed_args.checkpoint_path)
    output_dir = os.path.realpath(parsed_args.output_dir)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError('Full checkpoint does not exist: {}'.format(checkpoint_path))
    _reject_existing_output(output_dir)

    batch_size = _integer(parsed_args.batch_size, 'batch_size', 1)
    num_workers = _integer(parsed_args.num_workers, 'num_workers', 0)
    topk_ratio = _fixed_float(
        parsed_args.tg_memory_topk_ratio, 'tg_memory_topk_ratio', PROTOCOL['memory_topk_ratio']
    )
    fusion_lambda = _fixed_float(
        parsed_args.tg_memory_fusion_lambda,
        'tg_memory_fusion_lambda',
        PROTOCOL['memory_fusion_lambda'],
    )
    route_threshold = _fixed_float(
        parsed_args.tg_memory_route_margin_threshold,
        'tg_memory_route_margin_threshold',
        PROTOCOL['route_margin_threshold'],
    )
    min_members = _fixed_integer(
        parsed_args.tg_memory_min_class_members,
        'tg_memory_min_class_members',
        PROTOCOL['memory_min_class_members'],
    )

    checkpoint_metadata = load_train_checkpoint_metadata(checkpoint_path)
    profile = _resolve_profile(parsed_args, checkpoint_metadata)
    data_path = os.path.realpath(parsed_args.data_path or profile.default_data_path)
    if not os.path.isdir(data_path):
        raise FileNotFoundError('dataset directory does not exist: {}'.format(data_path))
    registries = {
        'full': _load_registry(parsed_args.full_pseudoclass_dir, profile, 'full'),
        'h_raw_e': _load_registry(parsed_args.raw_pseudoclass_dir, profile, 'h_raw_e'),
    }
    shared = validate_shared_registries(registries, num_classes=len(profile.item_list))

    device_index = _integer(parsed_args.gpu, 'gpu', 0)
    if not torch.cuda.is_available():
        raise RuntimeError('paired calibrated evaluation requires CUDA; refusing CPU fallback')
    if device_index >= torch.cuda.device_count():
        raise ValueError(
            'gpu index {} is unavailable (device_count={})'.format(
                device_index, torch.cuda.device_count()
            )
        )
    device = torch.device('cuda:{}'.format(device_index))
    model, _, loaded_metadata = load_model_from_train_checkpoint(checkpoint_path, device)
    train_data_list, test_data_list = build_datasets(
        data_path,
        profile.item_list,
        image_size=checkpoint_metadata['image_size'],
        crop_size=checkpoint_metadata['crop_size'],
    )
    loaders = rebuild_pruned_train_loaders(
        train_data_list,
        shared['retained_index_map'],
        data_root=data_path,
        item_list=profile.item_list,
        batch_size=batch_size,
        num_workers=num_workers,
        diag_batch_size=batch_size,
        diag_num_workers=num_workers,
    )
    max_patches = _integer(
        parsed_args.tg_mem_max_patches_per_class
        if parsed_args.tg_mem_max_patches_per_class is not None
        else _checkpoint_args_get(checkpoint_metadata['args'], 'tg_mem_max_patches_per_class', 20000),
        'tg_mem_max_patches_per_class',
        0,
    )
    chunk_size = _integer(
        parsed_args.tg_mem_chunk_size
        if parsed_args.tg_mem_chunk_size is not None
        else _checkpoint_args_get(checkpoint_metadata['args'], 'tg_mem_chunk_size', 4096),
        'tg_mem_chunk_size',
        1,
    )
    eval_args = SimpleNamespace(
        batch_size=batch_size,
        num_workers=num_workers,
        diag_max_ratio=float(checkpoint_metadata['diag_max_ratio']),
        diag_resize_mask=int(checkpoint_metadata['diag_resize_mask']),
        tg_memory_topk_ratio=topk_ratio,
        tg_memory_fusion_lambda=fusion_lambda,
        tg_memory_route_margin_threshold=route_threshold,
        tg_memory_min_class_members=min_members,
        tg_mem_chunk_size=chunk_size,
        tg_mem_max_patches_per_class=max_patches,
    )

    cache_started = time.time()
    train_cache = extract_shared_train_cache(
        model, loaders['train_eval_dataloader'], shared, registries, device
    )
    print(
        'paired train cache: {} retained samples in {:.1f}s'.format(
            len(train_cache), time.time() - cache_started
        ),
        flush=True,
    )
    memory_started = time.time()
    memory_systems = {
        variant: build_memory_system_from_cache(registries[variant], train_cache, max_patches)
        for variant in VARIANTS
    }
    print(
        'paired memory systems built in {:.1f}s'.format(time.time() - memory_started),
        flush=True,
    )
    del train_cache
    results, equivalence_audit = run_paired_test_evaluation(
        model, test_data_list, profile.item_list, memory_systems, eval_args, device
    )
    paired_scores = build_paired_score_audit(results)
    metric_rows = _metric_audit_rows(results)
    registry_rows = _registry_audit_rows(registries, shared)

    checkpoint_stat = os.stat(checkpoint_path)
    provenance = {
        'schema_version': 1,
        'evaluator': 'dinomaly_tailguard_paired_memory_eval',
        'no_training': True,
        'single_pass_contract': {
            'shared_full_checkpoint': True,
            'shared_retained_train_set': True,
            'one_train_encoder_pass': True,
            'one_test_joint_model_pass': True,
            'equivalence_audit': equivalence_audit,
        },
        'started_at_utc': started_at,
        'finished_at_utc': _utc_now(),
        'protocol': dict(PROTOCOL),
        'runtime_parameters': {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'diag_max_ratio': eval_args.diag_max_ratio,
            'diag_resize_mask': eval_args.diag_resize_mask,
            'memory_chunk_size': chunk_size,
            'max_patches_per_class': max_patches,
        },
        'checkpoint': {
            'path': checkpoint_path,
            'sha256': _sha256_file(checkpoint_path),
            'size_bytes': int(checkpoint_stat.st_size),
            'mtime_ns': int(checkpoint_stat.st_mtime_ns),
            'iteration': loaded_metadata['iteration'],
            'encoder_name': loaded_metadata['encoder_name'],
            'image_size': loaded_metadata['image_size'],
            'crop_size': loaded_metadata['crop_size'],
            'target_layers': loaded_metadata['target_layers'],
            'fuse_layer_encoder': loaded_metadata['fuse_layer_encoder'],
            'fuse_layer_decoder': loaded_metadata['fuse_layer_decoder'],
            'reconstruction_summary': loaded_metadata['final_eval_summary'],
        },
        'dataset': profile_provenance(profile, data_path),
        'shared_retained_set': {
            key: value for key, value in shared.items() if key != 'retained_index_map'
        },
        'retained_count_by_class': {
            str(key): len(value) for key, value in shared['retained_index_map'].items()
        },
        'registry_audit': registry_rows,
        'environment': _environment_provenance(device),
    }
    summary = {
        'schema_version': 1,
        'protocol': dict(PROTOCOL),
        'no_training': True,
        'full': results['full']['summary'],
        'h_raw_e': results['h_raw_e']['summary'],
        'full_minus_h_raw_e': {
            key: results['full']['summary'][key] - results['h_raw_e']['summary'][key]
            for key in METRIC_KEYS
        },
        'reconstruction_consistency': {
            key: (
                results['full']['metrics_by_mode']['recon']['summary'][key]
                - results['h_raw_e']['metrics_by_mode']['recon']['summary'][key]
            )
            for key in METRIC_KEYS
        },
        'provenance_file': 'paired_provenance.json',
    }
    if any(abs(value) > 1e-12 for value in summary['reconstruction_consistency'].values()):
        raise RuntimeError('paired registries produced different reconstruction metrics')

    temp_dir = '{}.inprogress-{}'.format(output_dir, os.getpid())
    if os.path.exists(temp_dir):
        raise FileExistsError('temporary output already exists: {}'.format(temp_dir))
    os.makedirs(temp_dir)
    try:
        for variant in VARIANTS:
            variant_dir = os.path.join(temp_dir, variant)
            variant_metadata = {
                'variant': variant,
                'paired_protocol': dict(PROTOCOL),
                'no_training': True,
                'provenance_file': '../paired_provenance.json',
                'registry_dir': registries[variant]['dir'],
                'shared_checkpoint_sha256': provenance['checkpoint']['sha256'],
                'shared_retained_identity_sha256': shared['stable_identity_sha256'],
            }
            save_tailguard_memory_artifacts(
                variant_dir,
                memory_systems[variant],
                results[variant]['score_df'],
                results[variant]['per_class_metrics'],
                results[variant]['summary'],
                variant_metadata,
                metrics_by_mode=results[variant]['metrics_by_mode'],
            )
        paired_scores.to_csv(os.path.join(temp_dir, 'paired_eval_scores.csv'), index=False)
        pd.DataFrame(metric_rows).to_csv(
            os.path.join(temp_dir, 'paired_metric_comparison.csv'), index=False
        )
        pd.DataFrame(registry_rows).to_csv(
            os.path.join(temp_dir, 'paired_registry_audit.csv'), index=False
        )
        per_class_rows = []
        for variant in VARIANTS:
            for mode, payload in results[variant]['metrics_by_mode'].items():
                for row in payload['per_class_metrics']:
                    per_class_rows.append({'variant': variant, 'mode': mode, **row})
        pd.DataFrame(per_class_rows).to_csv(
            os.path.join(temp_dir, 'paired_per_class_metrics.csv'), index=False
        )
        _write_json(os.path.join(temp_dir, 'paired_provenance.json'), provenance)
        _write_json(os.path.join(temp_dir, 'paired_summary.json'), summary)
        os.rename(temp_dir, output_dir)
    except Exception:
        raise
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False))
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description='Paired Full versus raw-tail memory evaluation without model training'
    )
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--full_pseudoclass_dir', required=True)
    parser.add_argument('--raw_pseudoclass_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--dataset_profile', choices=dataset_profile_names(), default=None)
    parser.add_argument('--data_path', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--tg_memory_topk_ratio', type=float, default=PROTOCOL['memory_topk_ratio'])
    parser.add_argument('--tg_memory_fusion_lambda', type=float, default=PROTOCOL['memory_fusion_lambda'])
    parser.add_argument(
        '--tg_memory_route_margin_threshold',
        type=float,
        default=PROTOCOL['route_margin_threshold'],
    )
    parser.add_argument(
        '--tg_memory_min_class_members',
        type=int,
        default=PROTOCOL['memory_min_class_members'],
    )
    parser.add_argument('--tg_mem_chunk_size', type=int, default=None)
    parser.add_argument('--tg_mem_max_patches_per_class', type=int, default=None)
    return parser


if __name__ == '__main__':
    paired_evaluate(build_parser().parse_args())
