"""Paired inference-only evaluation of Full, Routing Shield, and tri-state P.

The evaluator consumes a disagreement-only tri-state registry produced by
``build_tailguard_tri_state_registry.py``.  Full, Full+Shield, and tri-state P
share one checkpoint, retained train set, train feature cache, and test model
forward.  No model parameter is updated.
"""

import argparse
import glob
import json
import os
import time
from types import SimpleNamespace

import numpy as np

# Keep the isolated replay compatible with the legacy Dinomaly environment.
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
    _joint_test_forward_once,
    _load_registry,
    _metrics_for_class,
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
from dinomaly_tailguard_routing_shield_eval import (
    PROTOCOL,
    _fixed_negative_infinity,
    apply_routing_shield,
    build_routing_shield_from_cache,
    build_shield_routing_cache,
    load_attached_samples,
)
from dinomaly_train_base import (
    build_datasets,
    load_model_from_train_checkpoint,
    load_train_checkpoint_metadata,
    rebuild_pruned_train_loaders,
)
from one_shot_memory import resize_memory_map
from tailguard_dataset_profiles import dataset_profile_names, profile_provenance
from tailguard_memory import route_evidence_mask
from utils import compute_image_level_scores, get_gaussian_kernel


MODES = ('recon', 'full', 'full_shield', 'tri_state')
TRI_STATE_FILES = {
    'sample_roles': 'tri_state_sample_roles.csv',
    'routing_classes': 'tri_state_routing_classes.csv',
    'routing_members': 'tri_state_routing_members.csv',
    'memory_members': 'tri_state_memory_members.csv',
    'summary': 'tri_state_registry_summary.json',
}


def _require_columns(frame, required, name):
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError('{} is missing required columns: {}'.format(name, missing))


def _boolean_column(series, name):
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {'true': True, 'false': False, '1': True, '0': False}
    invalid = sorted(set(normalized).difference(mapping))
    if invalid:
        raise ValueError('{} contains invalid boolean values: {}'.format(name, invalid))
    return normalized.map(mapping).astype(bool)


def load_disagreement_tri_state_registry(directory, full_registry):
    """Load and verify the no-threshold, train-only tri-state registry."""
    root = os.path.realpath(directory)
    paths = {key: os.path.join(root, filename) for key, filename in TRI_STATE_FILES.items()}
    for path in paths.values():
        if not os.path.isfile(path):
            raise FileNotFoundError('tri-state registry artifact does not exist: {}'.format(path))
    roles = pd.read_csv(paths['sample_roles'])
    classes = pd.read_csv(paths['routing_classes'])
    routing_members = pd.read_csv(paths['routing_members'])
    memory_members = pd.read_csv(paths['memory_members'])
    with open(paths['summary'], 'r', encoding='utf-8') as file:
        summary = json.load(file)

    _require_columns(
        roles,
        {
            'sample_idx', 'class_id', 'base_idx', 'routing_role',
            'memory_eligible', 'cross_view_status', 'low_margin',
        },
        'tri-state sample roles',
    )
    _require_columns(
        classes,
        {
            'route_class_id', 'route_class_type', 'pseudo_class_type',
            'memory_eligible', 'num_routing_members', 'num_memory_members',
        },
        'tri-state routing classes',
    )
    _require_columns(
        routing_members,
        {
            'sample_idx', 'class_id', 'base_idx', 'route_class_id',
            'route_class_type',
        },
        'tri-state routing members',
    )
    _require_columns(
        memory_members,
        {
            'sample_idx', 'class_id', 'base_idx', 'route_class_id',
            'route_class_type',
        },
        'tri-state memory members',
    )
    for frame in (roles, classes, routing_members, memory_members):
        for column in set(frame.columns).intersection({
            'sample_idx', 'class_id', 'base_idx', 'route_class_id',
            'num_routing_members', 'num_memory_members',
        }):
            frame[column] = frame[column].astype(int)
    for frame in (roles, classes, routing_members, memory_members):
        for column in set(frame.columns).intersection({'memory_eligible', 'low_margin'}):
            frame[column] = _boolean_column(frame[column], column)

    policy = summary.get('policy', {})
    if policy.get('low_margin_threshold') is not None:
        raise ValueError('tri-state evaluation is frozen to disagreement-only (no margin threshold)')
    if not bool(policy.get('enable_auxiliary_memory', False)):
        raise ValueError('tri-state disagreement replay requires auxiliary memory')
    if not bool(summary.get('runtime_decision_is_train_only', False)):
        raise ValueError('tri-state registry does not declare train-only runtime evidence')
    if summary.get('oracle_columns_consumed') not in ([], None):
        raise ValueError('tri-state registry consumed oracle columns')
    if roles['low_margin'].any():
        raise ValueError('disagreement-only registry unexpectedly contains low-margin decisions')
    auxiliary_ids = set(roles.loc[
        roles['routing_role'] == 'auxiliary_tail', 'sample_idx'
    ].astype(int))
    disagreement_ids = set(roles.loc[
        roles['cross_view_status'] == 'disagrees', 'sample_idx'
    ].astype(int))
    if auxiliary_ids != disagreement_ids:
        raise ValueError('auxiliary memory must equal the cross-view disagreement set')

    full_members = full_registry['members']
    retained = full_members.set_index('sample_idx')[['class_id', 'base_idx']].sort_index()
    role_identity = roles.set_index('sample_idx')[['class_id', 'base_idx']].sort_index()
    if not role_identity.equals(retained):
        raise ValueError('tri-state roles do not share the Full retained identities')
    if roles['sample_idx'].duplicated().any():
        raise ValueError('tri-state roles contain duplicate sample_idx values')
    if memory_members['sample_idx'].duplicated().any():
        raise ValueError('tri-state patch-memory members are not unique')
    if set(routing_members['sample_idx'].astype(int)) != set(roles['sample_idx'].astype(int)):
        raise ValueError('tri-state routing does not cover every retained sample')

    actual_routing = routing_members.groupby('route_class_id').size().to_dict()
    expected_routing = classes.set_index('route_class_id')['num_routing_members'].to_dict()
    actual_memory = memory_members.groupby('route_class_id').size().to_dict()
    expected_memory = {
        int(row.route_class_id): int(row.num_memory_members)
        for row in classes.itertuples(index=False)
        if int(row.num_memory_members) > 0
    }
    if actual_routing != expected_routing or actual_memory != expected_memory:
        raise ValueError('tri-state class membership counts are inconsistent')
    shield_ids = set(classes.loc[classes['route_class_type'] == 'shield', 'route_class_id'])
    if set(memory_members['route_class_id']).intersection(shield_ids):
        raise ValueError('tri-state shields must not own patch memory')
    return {
        'dir': root,
        'paths': paths,
        'roles': roles.sort_values('sample_idx', kind='mergesort').reset_index(drop=True),
        'routing_classes': classes.sort_values('route_class_id', kind='mergesort').reset_index(drop=True),
        'routing_members': routing_members.sort_values(
            ['route_class_id', 'sample_idx'], kind='mergesort'
        ).reset_index(drop=True),
        'memory_members': memory_members.sort_values(
            ['route_class_id', 'sample_idx'], kind='mergesort'
        ).reset_index(drop=True),
        'summary': summary,
    }


def build_tri_state_cache_registry(full_registry, tri_state):
    """Mark the union of Full and tri-state memory samples for shared extraction."""
    members = full_registry['members'].copy()
    full_tail = set(members.loc[
        members['pseudo_class_type'].astype(str) == 'tail', 'sample_idx'
    ].astype(int))
    tri_tail = set(tri_state['memory_members']['sample_idx'].astype(int))
    tail_required = full_tail | tri_tail
    members['pseudo_class_id'] = 0
    members['pseudo_class_type'] = np.where(
        members['sample_idx'].astype(int).isin(tail_required), 'tail', 'head'
    )
    return {
        'variant': 'tri_state_cache_union',
        'members': members,
        'classes': pd.DataFrame(),
    }


def build_tri_state_memory_registry(tri_state):
    """Build routing prototypes without implicitly creating any patch bank."""
    classes = tri_state['routing_classes'].rename(columns={
        'route_class_id': 'pseudo_class_id',
        'num_routing_members': 'num_members',
    }).copy()
    members = tri_state['routing_members'].rename(columns={
        'route_class_id': 'pseudo_class_id',
    }).copy()
    # The shared constructor normally equates every tail routing member with a
    # patch-memory member.  Construct all routes as CLS-only heads first; the
    # explicit memory relation is applied in the next stage.
    classes['route_pseudo_class_type'] = classes['pseudo_class_type'].astype(str)
    classes['pseudo_class_type'] = 'head'
    type_lookup = classes.set_index('pseudo_class_id')['pseudo_class_type'].astype(str)
    members['pseudo_class_type'] = members['pseudo_class_id'].map(type_lookup)
    if members['pseudo_class_type'].isna().any():
        raise ValueError('tri-state routing members reference an unknown class')
    return {
        'variant': 'tri_state_disagreement_only',
        'members': members,
        'classes': classes,
    }


def apply_explicit_tri_state_memory_membership(memory_system, tri_state, train_cache, max_patches):
    """Rebuild patch banks from the explicit memory relation, never routing members."""
    from dinomaly_tailguard_paired_memory_eval import _allocate_patch_quotas, _sample_patches

    replacement = dict(memory_system)
    replacement['class_entries'] = {
        int(class_id): dict(entry)
        for class_id, entry in memory_system['class_entries'].items()
    }
    classes = tri_state['routing_classes'].set_index('route_class_id')
    memory_members = tri_state['memory_members']
    for route_class_id, entry in replacement['class_entries'].items():
        route_type = str(classes.loc[route_class_id, 'route_class_type'])
        explicit = memory_members.loc[
            memory_members['route_class_id'].astype(int) == int(route_class_id)
        ].sort_values('sample_idx', kind='mergesort')
        entry['num_routing_members'] = int(entry['num_members'])
        entry['num_memory_members'] = int(len(explicit))
        if route_type in {'head', 'shield'}:
            if len(explicit) != 0:
                raise ValueError('{} route unexpectedly owns explicit memory'.format(route_type))
            if entry.get('has_memory_bank') or 'features' in entry:
                raise ValueError('{} route unexpectedly received patch features'.format(route_type))
            entry['pseudo_class_type'] = 'head'
            continue
        if route_type not in {'open_tail', 'auxiliary_tail'}:
            raise ValueError('unknown tri-state route type: {}'.format(route_type))
        if len(explicit) == 0:
            raise ValueError('{} route has no explicit memory members'.format(route_type))
        keys = [
            (int(row.class_id), int(row.base_idx))
            for row in explicit.itertuples(index=False)
        ]
        records = [train_cache[key] for key in keys]
        if any('patches' not in record for record in records):
            raise ValueError('{} route is missing cached explicit patches'.format(route_type))
        dimensions = {int(record['patches'].shape[1]) for record in records}
        spatial_sizes = {record['spatial_size'] for record in records}
        if len(dimensions) != 1 or len(spatial_sizes) != 1:
            raise ValueError('{} explicit memory members have inconsistent patch shapes'.format(
                route_type
            ))
        capacities = [int(record['patches'].shape[0]) for record in records]
        quotas = _allocate_patch_quotas(capacities, max_patches)
        selected = [
            _sample_patches(record['patches'], quota)
            for record, quota in zip(records, quotas)
            if quota > 0
        ]
        if not selected:
            raise ValueError('{} explicit memory bank has no patches'.format(route_type))
        features = torch.cat(selected, dim=0).contiguous().cpu()
        entry.update({
            # The shared scorer uses num_members for its minimum-memory-support
            # gate, so tail routes must expose explicit memory support here.
            'num_members': int(len(explicit)),
            'pseudo_class_type': 'tail',
            'has_memory_bank': True,
            'features': features,
            'num_patches': int(features.shape[0]),
            'feature_dim': int(features.shape[1]),
            'spatial_size': next(iter(spatial_sizes)),
            'memory_member_sample_idx': explicit['sample_idx'].astype(int).tolist(),
            'member_patch_quotas': [
                {
                    'sample_idx': int(sample_idx),
                    'num_available_patches': int(capacity),
                    'num_selected_patches': int(quota),
                }
                for sample_idx, capacity, quota in zip(
                    explicit['sample_idx'].astype(int), capacities, quotas
                )
            ],
        })
    audit_tri_state_memory_system(replacement, tri_state)
    replacement['num_tail_memory_banks'] = sum(
        bool(entry.get('has_memory_bank'))
        for entry in replacement['class_entries'].values()
    )
    replacement['num_head_pseudo_classes'] = sum(
        entry['pseudo_class_type'] == 'head'
        for entry in replacement['class_entries'].values()
    )
    replacement['num_tail_pseudo_classes'] = sum(
        entry['pseudo_class_type'] == 'tail'
        for entry in replacement['class_entries'].values()
    )
    return replacement


def audit_tri_state_memory_system(memory_system, tri_state):
    """Fail if route roles and stored patch-bank ownership diverge."""
    classes = tri_state['routing_classes'].set_index('route_class_id')
    explicit_by_class = {
        int(route_class_id): set(group['sample_idx'].astype(int))
        for route_class_id, group in tri_state['memory_members'].groupby('route_class_id')
    }
    for route_class_id, entry in memory_system['class_entries'].items():
        route_type = str(classes.loc[int(route_class_id), 'route_class_type'])
        expected_compatible_type = str(classes.loc[int(route_class_id), 'pseudo_class_type'])
        if str(entry.get('pseudo_class_type')) != expected_compatible_type:
            raise ValueError('{} route has incompatible scorer type'.format(route_type))
        explicit_ids = explicit_by_class.get(int(route_class_id), set())
        stored_ids = set(int(value) for value in entry.get('memory_member_sample_idx', []))
        if route_type in {'head', 'shield'}:
            if entry.get('has_memory_bank') or entry.get('num_patches', 0) or explicit_ids:
                raise ValueError('{} route has a forbidden patch bank'.format(route_type))
            if 'features' in entry and int(entry['features'].shape[0]) > 0:
                raise ValueError('{} route stores forbidden patch features'.format(route_type))
        else:
            if not entry.get('has_memory_bank') or int(entry.get('num_patches', 0)) <= 0:
                raise ValueError('{} route is missing its patch bank'.format(route_type))
            if int(entry.get('num_members', -1)) != len(explicit_ids):
                raise ValueError('{} route support does not equal explicit memory membership'.format(
                    route_type
                ))
            if stored_ids != explicit_ids:
                raise ValueError('{} route patch members do not match the explicit relation'.format(
                    route_type
                ))
            quota_ids = {
                int(row['sample_idx']) for row in entry.get('member_patch_quotas', [])
            }
            if quota_ids != explicit_ids:
                raise ValueError('{} route patch quotas do not match explicit members'.format(
                    route_type
                ))
    return {
        'num_routes': int(len(memory_system['class_entries'])),
        'num_patch_banks': int(sum(
            bool(entry.get('has_memory_bank'))
            for entry in memory_system['class_entries'].values()
        )),
        'shield_banks': 0,
        'head_banks': 0,
    }


def _tri_state_route_types(result, tri_state):
    lookup = tri_state['routing_classes'].set_index('route_class_id')['route_class_type'].to_dict()
    return [lookup[int(value)] for value in result['predicted_pseudo_class_id'].tolist()]


def _append_prediction(storage, label, gt, score, anomaly_map):
    storage['gt_sp'].append(label.cpu())
    storage['pred_sp'].append(score.detach().cpu())
    storage['gt_px'].append(gt[:, 0].detach().cpu())
    storage['pred_px'].append(anomaly_map[:, 0].detach().cpu())


def _fuse(reconstruction_map, reconstruction_score, memory_result, args, device):
    applied = route_evidence_mask(
        memory_result['route_eligible'],
        memory_result['similarity_margin'].to(device),
        args.tg_memory_route_margin_threshold,
    )
    memory_map = resize_memory_map(
        memory_result['memory_patch_maps'], reconstruction_map.shape[-2:]
    ).unsqueeze(1)
    score = torch.where(
        applied,
        reconstruction_score + args.tg_memory_fusion_lambda * memory_result['memory_scores'],
        reconstruction_score,
    )
    anomaly_map = torch.where(
        applied[:, None, None, None],
        reconstruction_map + args.tg_memory_fusion_lambda * memory_map,
        reconstruction_map,
    )
    return score, anomaly_map, applied


@torch.no_grad()
def run_tri_state_evaluation(model,
                             test_data_list,
                             item_list,
                             full_memory_system,
                             shield_system,
                             tri_memory_system,
                             tri_state,
                             args,
                             device):
    gaussian = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    full_routing = _routing_cache(full_memory_system, device)
    shield_routing = build_shield_routing_cache(full_memory_system, shield_system, device)
    tri_routing = _routing_cache(tri_memory_system, device)
    memory_caches = {'full': {}, 'tri_state': {}}
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
                    patches, spatial_size, cls, full_memory_system, args, device,
                    full_routing, memory_caches['full'],
                )
                shield = apply_routing_shield(cls, full, shield_routing)
                tri = score_cached_features(
                    patches, spatial_size, cls, tri_memory_system, args, device,
                    tri_routing, memory_caches['tri_state'],
                )
                tri_route_types = _tri_state_route_types(tri, tri_state)

                full_score, full_map, full_applied = _fuse(
                    reconstruction_map, reconstruction_score, full, args, device
                )
                shield_score, shield_map, shield_applied = _fuse(
                    reconstruction_map, reconstruction_score, shield, args, device
                )
                tri_score, tri_map, tri_applied = _fuse(
                    reconstruction_map, reconstruction_score, tri, args, device
                )
                _append_prediction(
                    predictions['recon'], label, gt, reconstruction_score, reconstruction_map
                )
                _append_prediction(predictions['full'], label, gt, full_score, full_map)
                _append_prediction(
                    predictions['full_shield'], label, gt, shield_score, shield_map
                )
                _append_prediction(predictions['tri_state'], label, gt, tri_score, tri_map)

                for index in range(images.shape[0]):
                    shield_id = int(shield['shield_id'][index].item())
                    shield_meta = shield_system['metadata'][shield_id] if shield_id >= 0 else None
                    score_rows.append({
                        'class_id': int(class_id),
                        'class_name': str(class_name),
                        'img_path': str(img_paths[index]),
                        'label': int(label[index].item()),
                        'recon_score': float(reconstruction_score[index].item()),
                        'full_final_score': float(full_score[index].item()),
                        'full_memory_score': float(full['memory_scores'][index].item()),
                        'full_mem_applied': int(full_applied[index].item()),
                        'full_route_class_id': int(full['predicted_pseudo_class_id'][index].item()),
                        'full_route_type': full['predicted_pseudo_class_type'][index],
                        'shield_final_score': float(shield_score[index].item()),
                        'shield_mem_applied': int(shield_applied[index].item()),
                        'shield_win': int(shield['shield_win'][index].item()),
                        'shield_source_sample_idx': (
                            -1 if shield_meta is None else int(shield_meta['sample_idx'])
                        ),
                        'full_memory_blocked_by_shield': int(
                            bool(full_applied[index].item())
                            and bool(shield['shield_win'][index].item())
                        ),
                        'tri_state_final_score': float(tri_score[index].item()),
                        'tri_state_memory_score': float(tri['memory_scores'][index].item()),
                        'tri_state_mem_applied': int(tri_applied[index].item()),
                        'tri_state_route_class_id': int(
                            tri['predicted_pseudo_class_id'][index].item()
                        ),
                        'tri_state_route_type': tri_route_types[index],
                        'tri_state_top1_similarity': float(tri['top1_similarity'][index].item()),
                        'tri_state_similarity_margin': float(
                            tri['similarity_margin'][index].item()
                        ),
                        'tri_state_minus_full_score': float(
                            tri_score[index].item() - full_score[index].item()
                        ),
                        'tri_state_minus_shield_score': float(
                            tri_score[index].item() - shield_score[index].item()
                        ),
                    })
                sample_count += int(images.shape[0])
            metrics = _metrics_for_class(class_id, class_name, predictions, sample_count)
            for mode in MODES:
                per_class[mode].append(metrics[mode])
            print(
                'tri-state class [{}/{}] {}: {} samples, {:.1f}s'.format(
                    class_id + 1, len(item_list), class_name, sample_count,
                    time.time() - started,
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
    return pd.DataFrame(score_rows), metrics_by_mode, equivalence_audit


def build_tri_state_route_audit(scores):
    return scores.groupby(
        ['tri_state_route_class_id', 'tri_state_route_type'], sort=True
    ).agg(
        num_images=('img_path', 'size'),
        num_normal=('label', lambda values: int((values == 0).sum())),
        num_anomaly=('label', lambda values: int((values == 1).sum())),
        num_memory_applied=('tri_state_mem_applied', 'sum'),
    ).reset_index()


def _reject_existing_output(output_dir):
    if os.path.exists(output_dir):
        raise FileExistsError('output_dir must not already exist: {}'.format(output_dir))
    in_progress = sorted(glob.glob(output_dir + '.inprogress-*'))
    if in_progress:
        raise FileExistsError('tri-state evaluation has in-progress output: {}'.format(in_progress[0]))


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
    full_registry = _load_registry(parsed_args.full_pseudoclass_dir, profile, 'full')
    attached, attached_path = load_attached_samples(
        parsed_args.tail_attached_samples_csv, full_registry
    )
    tri_state = load_disagreement_tri_state_registry(
        parsed_args.tri_state_registry_dir, full_registry
    )
    cache_registry = build_tri_state_cache_registry(full_registry, tri_state)
    registries = {'full': full_registry, 'h_raw_e': cache_registry}
    shared = validate_shared_registries(registries, num_classes=len(profile.item_list))

    gpu = _integer(parsed_args.gpu, 'gpu', 0)
    if not torch.cuda.is_available():
        raise RuntimeError('tri-state evaluation requires CUDA; refusing CPU fallback')
    if gpu >= torch.cuda.device_count():
        raise ValueError('gpu index {} is unavailable (device_count={})'.format(
            gpu, torch.cuda.device_count()
        ))
    device = torch.device('cuda:{}'.format(gpu))
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
    full_memory_system = build_memory_system_from_cache(
        full_registry, cache, max_patches
    )
    shield_system = build_routing_shield_from_cache(attached, cache)
    tri_memory_system = build_memory_system_from_cache(
        build_tri_state_memory_registry(tri_state), cache, max_patches
    )
    tri_memory_system = apply_explicit_tri_state_memory_membership(
        tri_memory_system, tri_state, cache, max_patches
    )
    memory_ownership_audit = audit_tri_state_memory_system(tri_memory_system, tri_state)
    del cache
    scores, metrics_by_mode, equivalence = run_tri_state_evaluation(
        model,
        test_data_list,
        profile.item_list,
        full_memory_system,
        shield_system,
        tri_memory_system,
        tri_state,
        args,
        device,
    )
    route_audit = build_tri_state_route_audit(scores)

    mode_summaries = {
        mode: metrics_by_mode[mode]['summary'] for mode in MODES
    }
    summary = {
        'schema_version': 1,
        'evaluator': 'dinomaly_tailguard_tri_state_eval',
        'no_training': True,
        'protocol': dict(PROTOCOL),
        'metrics': mode_summaries,
        'tri_state_minus_full': {
            key: mode_summaries['tri_state'][key] - mode_summaries['full'][key]
            for key in METRIC_KEYS
        },
        'tri_state_minus_full_shield': {
            key: mode_summaries['tri_state'][key] - mode_summaries['full_shield'][key]
            for key in METRIC_KEYS
        },
        'routing': {
            'num_test_images': int(len(scores)),
            'num_full_shields': int(shield_system['num_shields']),
            'num_tri_state_routes': int(tri_memory_system['num_pseudo_classes']),
            'num_tri_state_shields': int((
                tri_state['routing_classes']['route_class_type'] == 'shield'
            ).sum()),
            'num_auxiliary_tail_classes': int((
                tri_state['routing_classes']['route_class_type'] == 'auxiliary_tail'
            ).sum()),
            'num_auxiliary_memory_samples': int((
                tri_state['roles']['routing_role'] == 'auxiliary_tail'
            ).sum()),
            'num_tri_state_memory_applied': int(scores['tri_state_mem_applied'].sum()),
            'memory_ownership_audit': memory_ownership_audit,
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
            'directory': full_registry['dir'],
            'members_sha256': _sha256_file(full_registry['paths']['members']),
            'classes_sha256': _sha256_file(full_registry['paths']['classes']),
            'registry_sha256': _sha256_file(full_registry['paths']['registry']),
        },
        'full_attached_candidates': {
            'path': attached_path,
            'sha256': _sha256_file(attached_path),
            'num_samples': int(len(attached)),
        },
        'tri_state_registry': {
            'directory': tri_state['dir'],
            'files': {
                key: {'path': path, 'sha256': _sha256_file(path)}
                for key, path in tri_state['paths'].items()
            },
            'summary': tri_state['summary'],
        },
        'tri_state_memory_ownership_audit': memory_ownership_audit,
        'shared_retained_set': {
            key: value for key, value in shared.items() if key != 'retained_index_map'
        },
        'environment': _environment_provenance(device),
    }

    temp_dir = '{}.inprogress-{}'.format(output_dir, os.getpid())
    os.makedirs(temp_dir)
    scores.to_csv(os.path.join(temp_dir, 'tri_state_scores.csv'), index=False)
    route_audit.to_csv(os.path.join(temp_dir, 'tri_state_route_audit.csv'), index=False)
    per_class_rows = []
    for mode in MODES:
        for row in metrics_by_mode[mode]['per_class_metrics']:
            per_class_rows.append({'mode': mode, **row})
    pd.DataFrame(per_class_rows).to_csv(
        os.path.join(temp_dir, 'tri_state_per_class_metrics.csv'), index=False
    )
    _write_json(os.path.join(temp_dir, 'tri_state_eval_summary.json'), summary)
    _write_json(os.path.join(temp_dir, 'tri_state_eval_provenance.json'), provenance)
    os.rename(temp_dir, output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description='Paired Full, Routing Shield, and disagreement-only tri-state evaluation'
    )
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--full_pseudoclass_dir', required=True)
    parser.add_argument('--tail_attached_samples_csv', required=True)
    parser.add_argument('--tri_state_registry_dir', required=True)
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
