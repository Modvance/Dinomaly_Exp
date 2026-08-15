"""Paired inference-only evaluation of Full TailGuard with Routing Shield.

Routing Shield retains one CLS prototype for every RGD-attached tail candidate
without retaining any of its patch features.  The prototypes join the Full
pseudo-class router; a shield win rejects memory fusion and falls back to the
unchanged reconstruction score.
"""

import argparse
import glob
import json
import math
import os
import time
from types import SimpleNamespace

import numpy as np

# The original Dinomaly environment combines NumPy 1.24 with legacy SciPy and
# scikit-learn releases that still import these aliases.
for _name, _value in {
    'int': int,
    'float': float,
    'bool': bool,
    'object': object,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)
if 'typeDict' not in np.__dict__:
    np.typeDict = np.sctypeDict

import pandas as pd
import torch
import torch.nn.functional as F

from dinomaly_tailguard_paired_memory_eval import (
    METRIC_KEYS,
    _checkpoint_args_get,
    _environment_provenance,
    _fixed_float,
    _fixed_integer,
    _integer,
    _json_safe,
    _joint_test_forward_once,
    _load_registry,
    _metrics_for_class,
    _normalize_rows,
    _resolve_profile,
    _routing_cache,
    _sha256_file,
    _utc_now,
    _write_json,
    build_memory_system_from_cache,
    extract_shared_train_cache,
    score_cached_features,
    validate_joint_forward_equivalence,
    validate_shared_registries,
)
from dinomaly_train_base import (
    build_datasets,
    load_model_from_train_checkpoint,
    load_train_checkpoint_metadata,
    rebuild_pruned_train_loaders,
)
from one_shot_memory import resize_memory_map
from tailguard_dataset_profiles import (
    dataset_profile_names,
    profile_provenance,
)
from tailguard_memory import route_evidence_mask
from utils import compute_image_level_scores, get_gaussian_kernel


PROTOCOL = {
    'memory_topk_ratio': 0.05,
    'memory_fusion_lambda': 1.0,
    'route_margin_threshold': float('-inf'),
    'memory_min_class_members': 1,
}
MODES = ('recon', 'full', 'full_shield')


def _fixed_negative_infinity(value, name):
    numeric = float(value)
    if numeric != float('-inf'):
        raise ValueError('{} is frozen at -inf: {}'.format(name, value))
    return numeric


def load_attached_samples(path, full_registry):
    """Load T_aff and verify every shield source is a retained Full sample."""
    attached_path = os.path.realpath(path)
    if not os.path.isfile(attached_path):
        raise FileNotFoundError('T_aff artifact does not exist: {}'.format(attached_path))
    attached = pd.read_csv(attached_path)
    required = {'sample_idx', 'class_id', 'base_idx'}
    missing = required.difference(attached.columns)
    if missing:
        raise ValueError('T_aff artifact is missing columns: {}'.format(sorted(missing)))
    for column in required:
        attached[column] = attached[column].astype(int)
    if attached.empty:
        raise ValueError('T_aff artifact contains no attached candidates')
    if attached['sample_idx'].duplicated().any() or attached.duplicated(['class_id', 'base_idx']).any():
        raise ValueError('T_aff artifact contains duplicate stable identities')
    if 'rgd_status' in attached.columns and not (attached['rgd_status'] == 'head_affiliated').all():
        unexpected = sorted(attached.loc[
            attached['rgd_status'] != 'head_affiliated', 'rgd_status'
        ].astype(str).unique())
        raise ValueError('T_aff artifact contains non-attached RGD states: {}'.format(unexpected))
    if 'tail_candidate' in attached.columns and not (attached['tail_candidate'].astype(int) == 1).all():
        raise ValueError('T_aff artifact contains non-tail candidates')

    members = full_registry['members']
    retained = {
        (int(row.class_id), int(row.base_idx)): int(row.sample_idx)
        for row in members.itertuples(index=False)
    }
    missing_identities = []
    remapped = []
    for row in attached.itertuples(index=False):
        key = (int(row.class_id), int(row.base_idx))
        if key not in retained:
            missing_identities.append(key)
        elif retained[key] != int(row.sample_idx):
            remapped.append((key, int(row.sample_idx), retained[key]))
    if missing_identities or remapped:
        raise ValueError(
            'T_aff does not match the retained Full registry (missing={}, remapped={})'.format(
                missing_identities[:10], remapped[:10]
            )
        )
    return attached.sort_values('sample_idx', kind='mergesort').reset_index(drop=True), attached_path


def build_routing_shield_from_cache(attached, train_cache):
    """Build per-sample CLS-only shields; patch tensors are never copied."""
    prototypes = []
    rows = []
    for shield_id, row in enumerate(attached.itertuples(index=False)):
        key = (int(row.class_id), int(row.base_idx))
        if key not in train_cache:
            raise ValueError('T_aff identity is absent from the shared train cache: {}'.format(key))
        record = train_cache[key]
        cls = record.get('cls')
        if cls is None or cls.ndim != 1 or not torch.isfinite(cls).all():
            raise ValueError('T_aff identity has an invalid CLS embedding: {}'.format(key))
        prototypes.append(cls.float().cpu().contiguous())
        metadata = {
            'shield_id': int(shield_id),
            'sample_idx': int(row.sample_idx),
            'class_id': int(row.class_id),
            'base_idx': int(row.base_idx),
            'has_patch_memory': False,
        }
        for column in ('sample_key', 'img_path', 'best_group_id', 'relative_group_dominance',
                       'rgd_threshold', 'rgd_status'):
            if hasattr(row, column):
                metadata[column] = getattr(row, column)
        rows.append(metadata)
    prototype_matrix = _normalize_rows(torch.stack(prototypes, dim=0)).cpu().contiguous()
    return {
        'schema_version': 1,
        'prototype_type': 'cls_only',
        'num_shields': len(rows),
        'prototypes': prototype_matrix,
        'metadata': rows,
        'has_patch_memory': False,
    }


def build_shield_routing_cache(memory_system, shield_system, device):
    base = _routing_cache(memory_system, device)
    shield_prototypes = _normalize_rows(shield_system['prototypes']).to(device)
    return {
        'base': base,
        'prototypes': torch.cat([base['prototypes'], shield_prototypes], dim=0),
        'num_base_routes': len(base['class_ids']),
        'shield_metadata': list(shield_system['metadata']),
    }


def apply_routing_shield(cls_embeddings, full_memory, routing_cache):
    """Reject Full memory only when an added CLS shield wins joint routing."""
    similarities = cls_embeddings @ routing_cache['prototypes'].T
    positions = similarities.argmax(dim=1)
    top1 = similarities.gather(1, positions[:, None])[:, 0]
    if similarities.shape[1] > 1:
        top2 = torch.topk(similarities, k=2, dim=1).values[:, 1]
    else:
        top2 = torch.full_like(top1, float('nan'))
    num_base = int(routing_cache['num_base_routes'])
    shield_wins = positions >= num_base

    memory_scores = torch.where(
        shield_wins,
        torch.zeros_like(full_memory['memory_scores']),
        full_memory['memory_scores'],
    )
    memory_maps = torch.where(
        shield_wins[:, None, None],
        torch.zeros_like(full_memory['memory_patch_maps']),
        full_memory['memory_patch_maps'],
    )
    eligible = torch.where(
        shield_wins,
        torch.zeros_like(full_memory['route_eligible']),
        full_memory['route_eligible'],
    )

    route_kinds = []
    route_ids = []
    pseudo_class_ids = []
    pseudo_class_types = []
    shield_ids = []
    shield_sample_indices = []
    for index, position in enumerate(positions.tolist()):
        if position < num_base:
            expected_id = int(routing_cache['base']['class_ids'][position])
            observed_id = int(full_memory['predicted_pseudo_class_id'][index].item())
            if expected_id != observed_id:
                raise RuntimeError('joint router changed the winning Full pseudo-class')
            route_kinds.append('pseudo_class')
            route_ids.append(expected_id)
            pseudo_class_ids.append(expected_id)
            pseudo_class_types.append(full_memory['predicted_pseudo_class_type'][index])
            shield_ids.append(-1)
            shield_sample_indices.append(-1)
        else:
            shield_id = int(position - num_base)
            shield = routing_cache['shield_metadata'][shield_id]
            route_kinds.append('shield')
            route_ids.append(shield_id)
            pseudo_class_ids.append(-1)
            pseudo_class_types.append('shield')
            shield_ids.append(shield_id)
            shield_sample_indices.append(int(shield['sample_idx']))
    return {
        'memory_scores': memory_scores,
        'memory_patch_maps': memory_maps,
        'route_eligible': eligible,
        'route_kind': route_kinds,
        'route_id': torch.tensor(route_ids, dtype=torch.int64),
        'predicted_pseudo_class_id': torch.tensor(pseudo_class_ids, dtype=torch.int64),
        'predicted_pseudo_class_type': pseudo_class_types,
        'shield_id': torch.tensor(shield_ids, dtype=torch.int64),
        'shield_sample_idx': torch.tensor(shield_sample_indices, dtype=torch.int64),
        'shield_win': shield_wins.detach().cpu().to(torch.int64),
        'top1_similarity': top1.detach().cpu(),
        'second_similarity': top2.detach().cpu(),
        'similarity_margin': (top1 - top2).detach().cpu(),
    }


def _append_prediction(storage, label, gt, score, anomaly_map):
    storage['gt_sp'].append(label.cpu())
    storage['pred_sp'].append(score.detach().cpu())
    storage['gt_px'].append(gt[:, 0].detach().cpu())
    storage['pred_px'].append(anomaly_map[:, 0].detach().cpu())


@torch.no_grad()
def run_shield_evaluation(model, test_data_list, item_list, memory_system, shield_system,
                          args, device):
    gaussian = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    full_routing = _routing_cache(memory_system, device)
    shield_routing = build_shield_routing_cache(memory_system, shield_system, device)
    device_memory_cache = {}
    score_rows = []
    per_class = {mode: [] for mode in MODES}
    equivalence_audit = None
    was_training = model.training
    model.eval()
    try:
        for class_id, (class_name, test_data) in enumerate(zip(item_list, test_data_list)):
            started = time.time()
            loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            predictions = {
                mode: {'gt_sp': [], 'pred_sp': [], 'gt_px': [], 'pred_px': []}
                for mode in MODES
            }
            sample_count = 0
            for images, gt, label, img_paths in loader:
                images = images.to(device)
                gt = gt.to(device)
                if equivalence_audit is None:
                    equivalence_audit = validate_joint_forward_equivalence(
                        model, images[:1], gaussian, args.diag_resize_mask
                    )
                reconstruction_map, patches, spatial_size, cls = _joint_test_forward_once(
                    model, images, gaussian, args.diag_resize_mask
                )
                gt = F.interpolate(gt, size=args.diag_resize_mask, mode='nearest').bool()
                if gt.shape[1] > 1:
                    gt = torch.max(gt, dim=1, keepdim=True)[0]
                reconstruction_score = compute_image_level_scores(
                    reconstruction_map, max_ratio=args.diag_max_ratio
                )
                full = score_cached_features(
                    patches,
                    spatial_size,
                    cls,
                    memory_system,
                    args,
                    device,
                    full_routing,
                    device_memory_cache,
                )
                shield = apply_routing_shield(cls, full, shield_routing)

                full_map = resize_memory_map(
                    full['memory_patch_maps'], reconstruction_map.shape[-2:]
                ).unsqueeze(1)
                shield_map = resize_memory_map(
                    shield['memory_patch_maps'], reconstruction_map.shape[-2:]
                ).unsqueeze(1)
                full_applied = route_evidence_mask(
                    full['route_eligible'],
                    full['similarity_margin'].to(device),
                    args.tg_memory_route_margin_threshold,
                )
                shield_applied = route_evidence_mask(
                    shield['route_eligible'],
                    shield['similarity_margin'].to(device),
                    args.tg_memory_route_margin_threshold,
                )
                full_score = torch.where(
                    full_applied,
                    reconstruction_score + args.tg_memory_fusion_lambda * full['memory_scores'],
                    reconstruction_score,
                )
                shield_score = torch.where(
                    shield_applied,
                    reconstruction_score + args.tg_memory_fusion_lambda * shield['memory_scores'],
                    reconstruction_score,
                )
                full_fused_map = torch.where(
                    full_applied[:, None, None, None],
                    reconstruction_map + args.tg_memory_fusion_lambda * full_map,
                    reconstruction_map,
                )
                shield_fused_map = torch.where(
                    shield_applied[:, None, None, None],
                    reconstruction_map + args.tg_memory_fusion_lambda * shield_map,
                    reconstruction_map,
                )
                _append_prediction(predictions['recon'], label, gt, reconstruction_score, reconstruction_map)
                _append_prediction(predictions['full'], label, gt, full_score, full_fused_map)
                _append_prediction(predictions['full_shield'], label, gt, shield_score, shield_fused_map)

                for index in range(images.shape[0]):
                    shield_id = int(shield['shield_id'][index].item())
                    shield_meta = (
                        shield_system['metadata'][shield_id] if shield_id >= 0 else None
                    )
                    blocked = bool(full_applied[index].item()) and bool(shield['shield_win'][index].item())
                    score_rows.append({
                        'class_id': int(class_id),
                        'class_name': str(class_name),
                        'img_path': str(img_paths[index]),
                        'label': int(label[index].item()),
                        'recon_score': float(reconstruction_score[index].item()),
                        'full_memory_score': float(full['memory_scores'][index].item()),
                        'full_final_score': float(full_score[index].item()),
                        'full_route_eligible': int(full['route_eligible'][index].item()),
                        'full_mem_applied': int(full_applied[index].item()),
                        'full_predicted_pseudo_class_id': int(
                            full['predicted_pseudo_class_id'][index].item()
                        ),
                        'full_predicted_pseudo_class_type': full['predicted_pseudo_class_type'][index],
                        'full_top1_similarity': float(full['top1_similarity'][index].item()),
                        'full_similarity_margin': float(full['similarity_margin'][index].item()),
                        'shield_final_score': float(shield_score[index].item()),
                        'shield_route_kind': shield['route_kind'][index],
                        'shield_route_eligible': int(shield['route_eligible'][index].item()),
                        'shield_mem_applied': int(shield_applied[index].item()),
                        'shield_id': shield_id,
                        'shield_source_sample_idx': -1 if shield_meta is None else int(shield_meta['sample_idx']),
                        'shield_source_class_id': -1 if shield_meta is None else int(shield_meta['class_id']),
                        'shield_source_base_idx': -1 if shield_meta is None else int(shield_meta['base_idx']),
                        'shield_source_best_group_id': -1 if shield_meta is None else int(
                            shield_meta.get('best_group_id', -1)
                        ),
                        'shield_top1_similarity': float(shield['top1_similarity'][index].item()),
                        'shield_similarity_margin': float(shield['similarity_margin'][index].item()),
                        'shield_win': int(shield['shield_win'][index].item()),
                        'full_memory_blocked': int(blocked),
                        'shield_minus_full_score': float(
                            shield_score[index].item() - full_score[index].item()
                        ),
                    })
                sample_count += int(images.shape[0])
            metrics = _metrics_for_class(class_id, class_name, predictions, sample_count)
            for mode in MODES:
                per_class[mode].append(metrics[mode])
            print(
                'routing shield class [{}/{}] {}: {} samples, {:.1f}s'.format(
                    class_id + 1, len(item_list), class_name, sample_count, time.time() - started
                ),
                flush=True,
            )
    finally:
        if was_training:
            model.train()

    metrics_by_mode = {}
    for mode in MODES:
        frame = pd.DataFrame(per_class[mode])
        metrics_by_mode[mode] = {
            'summary': {key: float(frame[key].mean()) for key in METRIC_KEYS},
            'per_class_metrics': per_class[mode],
        }
    scores = pd.DataFrame(score_rows)
    return scores, metrics_by_mode, equivalence_audit


def build_route_audits(scores):
    class_rows = []
    for (class_id, class_name), frame in scores.groupby(['class_id', 'class_name'], sort=True):
        baseline_applied = int(frame['full_mem_applied'].sum())
        blocked = int(frame['full_memory_blocked'].sum())
        class_rows.append({
            'class_id': int(class_id),
            'class_name': str(class_name),
            'num_images': int(len(frame)),
            'num_shield_wins': int(frame['shield_win'].sum()),
            'shield_win_ratio': float(frame['shield_win'].mean()),
            'num_full_memory_applied': baseline_applied,
            'num_full_memory_blocked': blocked,
            'blocked_among_full_applied_ratio': (
                float(blocked / baseline_applied) if baseline_applied else 0.0
            ),
            'num_normal_memory_blocked': int(
                ((frame['label'] == 0) & (frame['full_memory_blocked'] == 1)).sum()
            ),
            'num_anomaly_memory_blocked': int(
                ((frame['label'] == 1) & (frame['full_memory_blocked'] == 1)).sum()
            ),
        })
    transitions = scores.groupby(
        [
            'full_predicted_pseudo_class_id',
            'full_predicted_pseudo_class_type',
            'shield_route_kind',
            'shield_id',
            'shield_source_sample_idx',
        ],
        dropna=False,
        sort=True,
    ).agg(
        num_images=('img_path', 'size'),
        num_normal=('label', lambda values: int((values == 0).sum())),
        num_anomaly=('label', lambda values: int((values == 1).sum())),
        num_full_memory_blocked=('full_memory_blocked', 'sum'),
    ).reset_index()
    return pd.DataFrame(class_rows), transitions


def _reject_existing_output(output_dir):
    if os.path.exists(output_dir):
        raise FileExistsError('output_dir must not already exist: {}'.format(output_dir))
    in_progress = sorted(glob.glob(output_dir + '.inprogress-*'))
    if in_progress:
        raise FileExistsError('routing shield evaluation has in-progress output: {}'.format(in_progress[0]))


def evaluate(parsed_args):
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
    route_threshold = _fixed_negative_infinity(
        parsed_args.tg_memory_route_margin_threshold,
        'tg_memory_route_margin_threshold',
    )
    min_members = _fixed_integer(
        parsed_args.tg_memory_min_class_members,
        'tg_memory_min_class_members',
        PROTOCOL['memory_min_class_members'],
    )

    metadata = load_train_checkpoint_metadata(checkpoint_path)
    profile = _resolve_profile(parsed_args, metadata)
    data_path = os.path.realpath(parsed_args.data_path or profile.default_data_path)
    if not os.path.isdir(data_path):
        raise FileNotFoundError('dataset directory does not exist: {}'.format(data_path))
    registry = _load_registry(parsed_args.full_pseudoclass_dir, profile, 'full')
    attached, attached_path = load_attached_samples(parsed_args.tail_attached_samples_csv, registry)
    # Reusing the same registry twice invokes the paired retained-set validator
    # and the shared-cache extractor without introducing a second variant.
    registries = {'full': registry, 'h_raw_e': registry}
    shared = validate_shared_registries(registries, num_classes=len(profile.item_list))

    gpu = _integer(parsed_args.gpu, 'gpu', 0)
    if not torch.cuda.is_available():
        raise RuntimeError('routing shield evaluation requires CUDA; refusing CPU fallback')
    if gpu >= torch.cuda.device_count():
        raise ValueError('gpu index {} is unavailable (device_count={})'.format(
            gpu, torch.cuda.device_count()
        ))
    device = torch.device('cuda:{}'.format(gpu))
    # torch 1.12/cu113 cannot run NVRTC initialization for Ada GPUs. Building
    # on CPU and then moving the loaded model avoids that initialization path.
    model, _, loaded_metadata = load_model_from_train_checkpoint(
        checkpoint_path, torch.device('cpu')
    )
    model = model.to(device)
    train_data_list, test_data_list = build_datasets(
        data_path,
        profile.item_list,
        image_size=metadata['image_size'],
        crop_size=metadata['crop_size'],
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
        else _checkpoint_args_get(metadata['args'], 'tg_mem_max_patches_per_class', 20000),
        'tg_mem_max_patches_per_class',
        0,
    )
    chunk_size = _integer(
        parsed_args.tg_mem_chunk_size
        if parsed_args.tg_mem_chunk_size is not None
        else _checkpoint_args_get(metadata['args'], 'tg_mem_chunk_size', 4096),
        'tg_mem_chunk_size',
        1,
    )
    args = SimpleNamespace(
        batch_size=batch_size,
        num_workers=num_workers,
        diag_max_ratio=float(metadata['diag_max_ratio']),
        diag_resize_mask=int(metadata['diag_resize_mask']),
        tg_memory_topk_ratio=topk_ratio,
        tg_memory_fusion_lambda=fusion_lambda,
        tg_memory_route_margin_threshold=route_threshold,
        tg_memory_min_class_members=min_members,
        tg_mem_chunk_size=chunk_size,
        tg_mem_max_patches_per_class=max_patches,
    )

    cache = extract_shared_train_cache(
        model, loaders['train_eval_dataloader'], shared, registries, device
    )
    memory_system = build_memory_system_from_cache(registry, cache, max_patches)
    shield_system = build_routing_shield_from_cache(attached, cache)
    del cache
    scores, metrics_by_mode, equivalence = run_shield_evaluation(
        model,
        test_data_list,
        profile.item_list,
        memory_system,
        shield_system,
        args,
        device,
    )
    route_by_class, transitions = build_route_audits(scores)

    full_summary = metrics_by_mode['full']['summary']
    shield_summary = metrics_by_mode['full_shield']['summary']
    reconstruction_summary = metrics_by_mode['recon']['summary']
    blocked = int(scores['full_memory_blocked'].sum())
    applied = int(scores['full_mem_applied'].sum())
    summary = {
        'schema_version': 1,
        'evaluator': 'dinomaly_tailguard_routing_shield_eval',
        'no_training': True,
        'protocol': dict(PROTOCOL),
        'reconstruction': reconstruction_summary,
        'full': full_summary,
        'full_shield': shield_summary,
        'full_shield_minus_full': {
            key: shield_summary[key] - full_summary[key] for key in METRIC_KEYS
        },
        'routing': {
            'num_test_images': int(len(scores)),
            'num_shield_prototypes': int(shield_system['num_shields']),
            'num_shield_wins': int(scores['shield_win'].sum()),
            'shield_win_ratio': float(scores['shield_win'].mean()),
            'num_full_memory_applied': applied,
            'num_full_memory_blocked': blocked,
            'blocked_among_full_applied_ratio': float(blocked / applied) if applied else 0.0,
        },
    }
    provenance = {
        'schema_version': 1,
        'started_at_utc': started_at,
        'finished_at_utc': _utc_now(),
        'no_training': True,
        'single_pass_contract': {
            'shared_full_checkpoint': True,
            'shared_retained_train_set': True,
            'one_train_encoder_pass': True,
            'one_test_joint_model_pass': True,
            'equivalence_audit': equivalence,
        },
        'checkpoint': {
            'path': checkpoint_path,
            'sha256': _sha256_file(checkpoint_path),
            'iteration': loaded_metadata['iteration'],
            'encoder_name': loaded_metadata['encoder_name'],
        },
        'dataset': profile_provenance(profile, data_path),
        'full_registry': {
            'directory': registry['dir'],
            'members_sha256': _sha256_file(registry['paths']['members']),
            'classes_sha256': _sha256_file(registry['paths']['classes']),
            'registry_sha256': _sha256_file(registry['paths']['registry']),
        },
        'tail_attached_samples': {
            'path': attached_path,
            'sha256': _sha256_file(attached_path),
            'num_samples': int(len(attached)),
        },
        'shared_retained_set': {
            key: value for key, value in shared.items() if key != 'retained_index_map'
        },
        'environment': _environment_provenance(device),
    }

    temp_dir = '{}.inprogress-{}'.format(output_dir, os.getpid())
    os.makedirs(temp_dir)
    try:
        scores.to_csv(os.path.join(temp_dir, 'routing_shield_scores.csv'), index=False)
        route_by_class.to_csv(
            os.path.join(temp_dir, 'routing_shield_route_by_class.csv'), index=False
        )
        transitions.to_csv(
            os.path.join(temp_dir, 'routing_shield_route_transitions.csv'), index=False
        )
        per_class_rows = []
        for mode in MODES:
            for row in metrics_by_mode[mode]['per_class_metrics']:
                per_class_rows.append({'mode': mode, **row})
        pd.DataFrame(per_class_rows).to_csv(
            os.path.join(temp_dir, 'routing_shield_per_class_metrics.csv'), index=False
        )
        pd.DataFrame(shield_system['metadata']).to_csv(
            os.path.join(temp_dir, 'routing_shield_prototypes.csv'), index=False
        )
        torch.save(
            {
                'schema_version': shield_system['schema_version'],
                'prototype_type': shield_system['prototype_type'],
                'prototypes': shield_system['prototypes'],
                'metadata': shield_system['metadata'],
                'has_patch_memory': False,
            },
            os.path.join(temp_dir, 'routing_shield_prototypes.pt'),
        )
        _write_json(os.path.join(temp_dir, 'routing_shield_summary.json'), summary)
        _write_json(os.path.join(temp_dir, 'routing_shield_provenance.json'), provenance)
        os.rename(temp_dir, output_dir)
    except Exception:
        raise
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False))
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description='Paired inference-only Full versus Full+Routing-Shield evaluation'
    )
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--full_pseudoclass_dir', required=True)
    parser.add_argument('--tail_attached_samples_csv', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--dataset_profile', choices=dataset_profile_names(), default=None)
    parser.add_argument('--data_path', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--tg_memory_topk_ratio', type=float, default=PROTOCOL['memory_topk_ratio'])
    parser.add_argument(
        '--tg_memory_fusion_lambda', type=float, default=PROTOCOL['memory_fusion_lambda']
    )
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
    evaluate(build_parser().parse_args())
