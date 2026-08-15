"""Rebuild TailGuard pseudo-class memory from a saved final checkpoint."""

import argparse
import json
import math
import os
from types import SimpleNamespace

import pandas as pd
import torch

from dinomaly_train_base import (
    _checkpoint_args_get,
    build_datasets,
    load_model_from_train_checkpoint,
    load_train_checkpoint_metadata,
    rebuild_pruned_train_loaders,
)
from tailguard_dataset_profiles import (
    MVTec_PROFILE,
    dataset_profile_names,
    get_dataset_profile,
    profile_provenance,
    validate_profile_contract,
)
from tailguard_artifacts import save_tailguard_memory_artifacts
from tailguard_gbps import _build_retained_index_map
from tailguard_memory import (
    build_tailguard_pseudoclass_memory_system,
    run_pseudoclass_memory_evaluation,
)
from tailguard_defaults import TAILGUARD_FINAL_DEFAULTS


def _resolve(value, metadata, name, default):
    if value is not None:
        return value
    checkpoint_args = metadata.get('args')
    missing = object()
    resolved = _checkpoint_args_get(checkpoint_args, name, missing)
    if resolved is missing and name.startswith('tg_'):
        legacy_name = 'tgb_' + name[len('tg_'):]
        resolved = _checkpoint_args_get(checkpoint_args, legacy_name, missing)
    if resolved is missing:
        resolved = default
    return default if resolved is None else resolved


def _integer_value(value, name, minimum):
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < minimum:
        raise ValueError('{} must be an integer >= {}: {}'.format(name, minimum, value))
    return int(numeric)


def _ratio_value(value, name):
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 < numeric <= 1.0:
        raise ValueError('{} must be in (0, 1]: {}'.format(name, value))
    return numeric


def _resolve_dataset_profile(parsed_args, checkpoint_metadata):
    checkpoint_profile_name = _checkpoint_args_get(checkpoint_metadata.get('args'), 'dataset_profile', None)
    if checkpoint_profile_name is not None:
        checkpoint_profile = get_dataset_profile(checkpoint_profile_name)
    else:
        checkpoint_profile = None

    if parsed_args.dataset_profile is not None:
        profile = get_dataset_profile(parsed_args.dataset_profile)
    elif checkpoint_profile is not None:
        profile = checkpoint_profile
    else:
        profile = MVTec_PROFILE
        print('checkpoint has no dataset profile; falling back to legacy MVTec profile')

    validate_profile_contract(
        profile,
        checkpoint_profile_name,
        _checkpoint_args_get(checkpoint_metadata.get('args'), 'dataset_item_list', None),
        'checkpoint',
    )
    return profile


def _validate_pseudoclass_profile(pseudoclass_dir, profile):
    registry_path = os.path.join(pseudoclass_dir, 'pseudo_class_registry.json')
    if not os.path.isfile(registry_path):
        print('pseudo-class registry has no provenance; allowing legacy rebuild')
        return
    with open(registry_path, 'r', encoding='utf-8') as file:
        registry = json.load(file)
    provenance = registry.get('dataset_provenance')
    if provenance is None:
        print('pseudo-class registry has no dataset provenance; allowing legacy rebuild')
        return
    if not isinstance(provenance, dict):
        raise ValueError('pseudo-class registry dataset_provenance must be an object')
    validate_profile_contract(
        profile,
        provenance.get('profile_name'),
        provenance.get('item_list'),
        'pseudo-class registry',
    )


def _retained_index_map(members_df, train_data_list):
    required_columns = {'class_id', 'base_idx'}
    missing_columns = required_columns.difference(members_df.columns)
    if missing_columns:
        raise ValueError(
            'pseudo-class members are missing stable identities: {}. Regenerate them with '
            'dinomaly_mvtec_tailguard_attachment_replay.py before rebuilding memory.'.format(
                ', '.join(sorted(missing_columns))
            )
        )
    members = members_df.copy()
    members['class_id'] = [_integer_value(value, 'class_id', 0) for value in members['class_id']]
    members['base_idx'] = [_integer_value(value, 'base_idx', 0) for value in members['base_idx']]
    if members.duplicated(['class_id', 'base_idx']).any():
        raise ValueError('pseudo-class members contain duplicate class_id/base_idx identities')
    for row in members[['class_id', 'base_idx']].itertuples(index=False):
        if row.class_id >= len(train_data_list):
            raise ValueError('pseudo-class members contain an unknown class_id: {}'.format(row.class_id))
        if row.base_idx >= len(train_data_list[row.class_id]):
            raise ValueError(
                'pseudo-class members contain an out-of-range base_idx for class {}: {}'.format(
                    row.class_id,
                    row.base_idx,
                )
            )
    all_classes_df = pd.DataFrame({'class_id': list(range(len(train_data_list)))})
    return _build_retained_index_map(members, all_samples_df=all_classes_df)


def rebuild_memory(parsed_args):
    checkpoint_path = os.path.realpath(parsed_args.checkpoint_path)
    pseudoclass_dir = os.path.realpath(parsed_args.pseudoclass_dir)
    output_dir = os.path.realpath(parsed_args.output_dir)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError('final model checkpoint does not exist: {}'.format(checkpoint_path))
    if not os.path.isdir(pseudoclass_dir):
        raise FileNotFoundError('pseudo-class directory does not exist: {}'.format(pseudoclass_dir))
    if os.path.exists(output_dir):
        raise FileExistsError('output_dir must not already exist: {}'.format(output_dir))

    checkpoint_metadata = load_train_checkpoint_metadata(checkpoint_path)
    profile = _resolve_dataset_profile(parsed_args, checkpoint_metadata)
    data_path = os.path.realpath(parsed_args.data_path or profile.default_data_path)
    if not os.path.isdir(data_path):
        raise FileNotFoundError('dataset directory does not exist: {}'.format(data_path))
    _validate_pseudoclass_profile(pseudoclass_dir, profile)
    members_df = pd.read_csv(os.path.join(pseudoclass_dir, 'pseudo_class_members.csv'))
    classes_df = pd.read_csv(os.path.join(pseudoclass_dir, 'pseudo_classes.csv'))
    gpu = _integer_value(parsed_args.gpu, 'gpu', 0)
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
    model, _, _ = load_model_from_train_checkpoint(checkpoint_path, device)
    dataset_provenance = profile_provenance(profile, data_path)
    checkpoint_provenance = {
        'dataset_provenance': dataset_provenance,
        'checkpoint_path': checkpoint_path,
        'iteration': checkpoint_metadata['iteration'],
        'encoder_name': checkpoint_metadata['encoder_name'],
        'image_size': int(checkpoint_metadata['image_size']),
        'crop_size': int(checkpoint_metadata['crop_size']),
        'final_eval_summary': checkpoint_metadata['final_eval_summary'],
    }
    batch_size = _integer_value(_resolve(parsed_args.batch_size, checkpoint_metadata, 'batch_size', 16), 'batch_size', 1)
    num_workers = _integer_value(_resolve(parsed_args.num_workers, checkpoint_metadata, 'num_workers', 4), 'num_workers', 0)
    diag_batch_size = _integer_value(
        _resolve(parsed_args.diag_batch_size, checkpoint_metadata, 'diag_batch_size', batch_size),
        'diag_batch_size',
        1,
    )
    diag_num_workers = _integer_value(
        _resolve(parsed_args.diag_num_workers, checkpoint_metadata, 'diag_num_workers', num_workers),
        'diag_num_workers',
        0,
    )
    memory_args = SimpleNamespace(
        batch_size=batch_size,
        num_workers=num_workers,
        diag_max_ratio=_ratio_value(
            _resolve(parsed_args.diag_max_ratio, checkpoint_metadata, 'diag_max_ratio', 0.01),
            'diag_max_ratio',
        ),
        diag_resize_mask=_integer_value(
            _resolve(parsed_args.diag_resize_mask, checkpoint_metadata, 'diag_resize_mask', 256),
            'diag_resize_mask',
            1,
        ),
        tg_memory_fusion_lambda=float(_resolve(
            parsed_args.tg_memory_fusion_lambda,
            checkpoint_metadata,
            'tg_memory_fusion_lambda',
            TAILGUARD_FINAL_DEFAULTS['tg_memory_fusion_lambda'],
        )),
        tg_memory_topk_ratio=_ratio_value(
            _resolve(
                parsed_args.tg_memory_topk_ratio,
                checkpoint_metadata,
                'tg_memory_topk_ratio',
                TAILGUARD_FINAL_DEFAULTS['tg_memory_topk_ratio'],
            ),
            'tg_memory_topk_ratio',
        ),
        tg_memory_route_margin_threshold=float(_resolve(
            parsed_args.tg_memory_route_margin_threshold,
            checkpoint_metadata,
            'tg_memory_route_margin_threshold',
            TAILGUARD_FINAL_DEFAULTS['tg_memory_route_margin_threshold'],
        )),
        tg_memory_min_class_members=_integer_value(
            _resolve(
                parsed_args.tg_memory_min_class_members,
                checkpoint_metadata,
                'tg_memory_min_class_members',
                TAILGUARD_FINAL_DEFAULTS['tg_memory_min_class_members'],
            ),
            'tg_memory_min_class_members',
            1,
        ),
        tg_mem_chunk_size=_integer_value(
            _resolve(
                parsed_args.tg_mem_chunk_size,
                checkpoint_metadata,
                'tg_mem_chunk_size',
                TAILGUARD_FINAL_DEFAULTS['tg_mem_chunk_size'],
            ),
            'tg_mem_chunk_size',
            1,
        ),
        tg_mem_max_patches_per_class=_integer_value(
            _resolve(
                parsed_args.tg_mem_max_patches_per_class,
                checkpoint_metadata,
                'tg_mem_max_patches_per_class',
                TAILGUARD_FINAL_DEFAULTS['tg_mem_max_patches_per_class'],
            ),
            'tg_mem_max_patches_per_class',
            0,
        ),
    )
    if not math.isfinite(memory_args.tg_memory_fusion_lambda) or memory_args.tg_memory_fusion_lambda < 0.0:
        raise ValueError('tg_memory_fusion_lambda must be finite and non-negative')
    if math.isnan(memory_args.tg_memory_route_margin_threshold) or memory_args.tg_memory_route_margin_threshold == float('inf'):
        raise ValueError('tg_memory_route_margin_threshold must not be NaN or +inf')
    del checkpoint_metadata
    train_data_list, test_data_list = build_datasets(
        data_path,
        profile.item_list,
        image_size=checkpoint_provenance['image_size'],
        crop_size=checkpoint_provenance['crop_size'],
    )
    loaders = rebuild_pruned_train_loaders(
        train_data_list,
        _retained_index_map(members_df, train_data_list),
        data_root=data_path,
        item_list=profile.item_list,
        batch_size=batch_size,
        num_workers=num_workers,
        diag_batch_size=diag_batch_size,
        diag_num_workers=diag_num_workers,
    )
    memory_system = build_tailguard_pseudoclass_memory_system(
        model,
        loaders['train_eval_dataloader'],
        device,
        members_df,
        classes_df,
        memory_args,
    )
    score_df, per_class_metrics, summary, metrics_by_mode = run_pseudoclass_memory_evaluation(
        model,
        test_data_list,
        profile.item_list,
        memory_system,
        memory_args,
        device,
    )
    os.makedirs(output_dir, exist_ok=True)
    artifacts = save_tailguard_memory_artifacts(
        output_dir,
        memory_system,
        score_df,
        per_class_metrics,
        summary,
        {
            'checkpoint_provenance': checkpoint_provenance,
            'dataset_provenance': dataset_provenance,
            'pseudoclass_dir': pseudoclass_dir,
            'data_path': data_path,
            'device': str(device),
            'batch_size': batch_size,
            'num_workers': num_workers,
            'diag_batch_size': diag_batch_size,
            'diag_num_workers': diag_num_workers,
            'diag_max_ratio': memory_args.diag_max_ratio,
            'diag_resize_mask': memory_args.diag_resize_mask,
            'num_pseudo_classes': int(memory_system['num_pseudo_classes']),
            'num_head_pseudo_classes': int(memory_system['num_head_pseudo_classes']),
            'num_tail_pseudo_classes': int(memory_system['num_tail_pseudo_classes']),
            'num_tail_memory_banks': int(memory_system['num_tail_memory_banks']),
            'tg_memory_fusion_lambda': memory_args.tg_memory_fusion_lambda,
            'tg_memory_topk_ratio': memory_args.tg_memory_topk_ratio,
            'tg_memory_route_margin_threshold': memory_args.tg_memory_route_margin_threshold,
            'tg_memory_min_class_members': memory_args.tg_memory_min_class_members,
            'tg_mem_chunk_size': memory_args.tg_mem_chunk_size,
            'tg_mem_max_patches_per_class': memory_args.tg_mem_max_patches_per_class,
        },
        metrics_by_mode=metrics_by_mode,
    )
    result = {
        'checkpoint_provenance': checkpoint_provenance,
        'dataset_provenance': dataset_provenance,
        'pseudoclass_dir': pseudoclass_dir,
        'data_path': data_path,
        'memory_artifacts': artifacts,
        'memory_eval_summary': summary,
    }
    with open(os.path.join(output_dir, 'memory_rebuild_summary.json'), 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    metric_keys = ('I-AUROC', 'I-AP', 'I-F1', 'P-AUROC', 'P-AP', 'P-F1', 'P-AUPRO')
    print(json.dumps({key: summary[key] for key in metric_keys}, indent=2, ensure_ascii=False))


def _add_hidden_legacy_alias(parser, option, dest, **kwargs):
    parser.add_argument(
        option,
        dest=dest,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
        **kwargs,
    )


def build_parser():
    parser = argparse.ArgumentParser(description='Build TailGuard final memory from a saved final model without retraining')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--pseudoclass_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset_profile', type=str, choices=dataset_profile_names(), default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--diag_batch_size', type=int, default=None)
    parser.add_argument('--diag_num_workers', type=int, default=None)
    parser.add_argument('--diag_max_ratio', type=float, default=None)
    parser.add_argument('--diag_resize_mask', type=int, default=None)
    parser.add_argument('--tg_memory_fusion_lambda', type=float, default=None)
    parser.add_argument('--tg_memory_topk_ratio', type=float, default=None)
    parser.add_argument('--tg_memory_route_margin_threshold', type=float, default=None)
    parser.add_argument('--tg_memory_min_class_members', type=int, default=None)
    parser.add_argument('--tg_mem_chunk_size', type=int, default=None)
    parser.add_argument('--tg_mem_max_patches_per_class', type=int, default=None)
    _add_hidden_legacy_alias(parser, '--tgb_memory_fusion_lambda', 'tg_memory_fusion_lambda', type=float)
    _add_hidden_legacy_alias(parser, '--tgb_memory_topk_ratio', 'tg_memory_topk_ratio', type=float)
    _add_hidden_legacy_alias(parser, '--tgb_mem_chunk_size', 'tg_mem_chunk_size', type=int)
    _add_hidden_legacy_alias(
        parser,
        '--tgb_mem_max_patches_per_class',
        'tg_mem_max_patches_per_class',
        type=int,
    )
    return parser


if __name__ == '__main__':
    rebuild_memory(build_parser().parse_args())
