"""Calibrate TailGuard E on label-free train-side synthetic pairs."""

import argparse
import json
import math
import os

import pandas as pd
import torch

from dinomaly_train_base import (
    _checkpoint_args_get,
    build_datasets,
    build_train_calibration_dataloader,
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
from tailguard_defaults import TAILGUARD_FINAL_DEFAULTS
from tailguard_gbps import _build_retained_index_map
from tailguard_memory import (
    build_tailguard_pseudoclass_memory_system,
    collect_tailguard_memory_calibration_records,
    select_tailguard_memory_calibration,
)


def _resolve(value, metadata, name, default):
    if value is not None:
        return value
    checkpoint_args = metadata.get('args')
    missing = object()
    resolved = _checkpoint_args_get(checkpoint_args, name, missing)
    if resolved is missing and name.startswith('tg_'):
        resolved = _checkpoint_args_get(checkpoint_args, 'tgb_' + name[len('tg_'):], missing)
    if resolved is missing:
        resolved = default
    return default if resolved is None else resolved


def _integer_value(value, name, minimum):
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < minimum:
        raise ValueError('{} must be an integer >= {}: {}'.format(name, minimum, value))
    return int(numeric)


def _resolve_dataset_profile(parsed_args, checkpoint_metadata):
    checkpoint_name = _checkpoint_args_get(checkpoint_metadata.get('args'), 'dataset_profile', None)
    if parsed_args.dataset_profile is not None:
        profile = get_dataset_profile(parsed_args.dataset_profile)
    elif checkpoint_name is not None:
        profile = get_dataset_profile(checkpoint_name)
    else:
        profile = MVTec_PROFILE
    validate_profile_contract(
        profile,
        checkpoint_name,
        _checkpoint_args_get(checkpoint_metadata.get('args'), 'dataset_item_list', None),
        'checkpoint',
    )
    return profile


def _validate_pseudoclass_profile(pseudoclass_dir, profile):
    registry_path = os.path.join(pseudoclass_dir, 'pseudo_class_registry.json')
    if not os.path.isfile(registry_path):
        return
    with open(registry_path, 'r', encoding='utf-8') as file:
        provenance = json.load(file).get('dataset_provenance')
    if provenance is not None:
        validate_profile_contract(
            profile,
            provenance.get('profile_name'),
            provenance.get('item_list'),
            'pseudo-class registry',
        )


def _retained_index_map(members_df, train_data_list):
    required = {'class_id', 'base_idx'}
    if not required.issubset(members_df.columns):
        raise ValueError('pseudo-class members require class_id and base_idx')
    members = members_df.copy()
    members['class_id'] = [_integer_value(value, 'class_id', 0) for value in members['class_id']]
    members['base_idx'] = [_integer_value(value, 'base_idx', 0) for value in members['base_idx']]
    for row in members[['class_id', 'base_idx']].itertuples(index=False):
        if row.class_id >= len(train_data_list) or row.base_idx >= len(train_data_list[row.class_id]):
            raise ValueError('pseudo-class member identity is outside the local training dataset')
    all_classes = pd.DataFrame({'class_id': list(range(len(train_data_list)))})
    return _build_retained_index_map(members, all_samples_df=all_classes)


def _float_list(value, name, minimum=None, maximum=None):
    values = []
    for raw in str(value).split(','):
        numeric = float(raw.strip())
        if not math.isfinite(numeric):
            raise ValueError('{} must contain only finite values'.format(name))
        if minimum is not None and numeric < minimum:
            raise ValueError('{} values must be >= {}'.format(name, minimum))
        if maximum is not None and numeric > maximum:
            raise ValueError('{} values must be <= {}'.format(name, maximum))
        values.append(numeric)
    if not values:
        raise ValueError('{} must not be empty'.format(name))
    return sorted(set(values))


def collect_calibration(parsed_args):
    checkpoint_path = os.path.realpath(parsed_args.checkpoint_path)
    pseudoclass_dir = os.path.realpath(parsed_args.pseudoclass_dir)
    output_dir = os.path.realpath(parsed_args.output_dir)
    if os.path.exists(output_dir):
        raise FileExistsError('output_dir must not already exist: {}'.format(output_dir))
    metadata = load_train_checkpoint_metadata(checkpoint_path)
    profile = _resolve_dataset_profile(parsed_args, metadata)
    data_path = os.path.realpath(parsed_args.data_path or profile.default_data_path)
    _validate_pseudoclass_profile(pseudoclass_dir, profile)
    members_df = pd.read_csv(os.path.join(pseudoclass_dir, 'pseudo_class_members.csv'))
    classes_df = pd.read_csv(os.path.join(pseudoclass_dir, 'pseudo_classes.csv'))
    device = torch.device(
        'cuda:{}'.format(_integer_value(parsed_args.gpu, 'gpu', 0))
        if torch.cuda.is_available() else 'cpu'
    )
    model, _, _ = load_model_from_train_checkpoint(checkpoint_path, device)
    batch_size = _integer_value(_resolve(parsed_args.batch_size, metadata, 'batch_size', 16), 'batch_size', 1)
    num_workers = _integer_value(_resolve(parsed_args.num_workers, metadata, 'num_workers', 4), 'num_workers', 0)
    topk_ratios = _float_list(
        parsed_args.tg_memory_topk_ratios,
        'tg_memory_topk_ratios',
        minimum=1e-12,
        maximum=1.0,
    )
    image_size = int(metadata['image_size'])
    crop_size = int(metadata['crop_size'])
    train_data_list, _ = build_datasets(data_path, profile.item_list, image_size=image_size, crop_size=crop_size)
    retained_index_map = _retained_index_map(members_df, train_data_list)
    loaders = rebuild_pruned_train_loaders(
        train_data_list,
        retained_index_map,
        data_root=data_path,
        item_list=profile.item_list,
        batch_size=batch_size,
        num_workers=num_workers,
        diag_batch_size=batch_size,
        diag_num_workers=num_workers,
    )
    memory_args = argparse.Namespace(
        batch_size=batch_size,
        num_workers=num_workers,
        diag_max_ratio=float(_resolve(None, metadata, 'diag_max_ratio', 0.01)),
        diag_resize_mask=int(_resolve(None, metadata, 'diag_resize_mask', 256)),
        tg_memory_topk_ratio=topk_ratios[0],
        tg_memory_min_class_members=_integer_value(
            parsed_args.tg_memory_min_class_members,
            'tg_memory_min_class_members',
            1,
        ),
        tg_mem_chunk_size=int(_resolve(None, metadata, 'tg_mem_chunk_size', 4096)),
        tg_mem_max_patches_per_class=int(_resolve(None, metadata, 'tg_mem_max_patches_per_class', 20000)),
    )
    memory_system = build_tailguard_pseudoclass_memory_system(
        model,
        loaders['train_eval_dataloader'],
        device,
        members_df,
        classes_df,
        memory_args,
    )
    calibration_loader = build_train_calibration_dataloader(
        loaders['train_data_list'],
        data_root=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        item_list=profile.item_list,
        corruption_grid=parsed_args.corruption_grid,
        corruption_min_ratio=parsed_args.corruption_min_ratio,
        corruption_max_ratio=parsed_args.corruption_max_ratio,
        corruption_strength=parsed_args.corruption_strength,
        corruption_seed=parsed_args.corruption_seed,
        sample_idx_maps={
            int(class_id): {
                int(row.base_idx): int(row.sample_idx)
                for row in members_df.loc[members_df['class_id'] == class_id].itertuples(index=False)
            }
            for class_id in members_df['class_id'].astype(int).unique()
        },
    )
    records = collect_tailguard_memory_calibration_records(
        model,
        calibration_loader,
        memory_system,
        memory_args,
        device,
        topk_ratios=topk_ratios,
        leave_one_out=not parsed_args.reference_preserving,
    )
    os.makedirs(output_dir)
    pd.DataFrame(records).to_csv(os.path.join(output_dir, 'calibration_records.csv'), index=False)
    payload = {
        'schema_version': 1,
        'protocol': (
            'label_free_train_synthetic_reference_preserving_v2'
            if parsed_args.reference_preserving
            else 'label_free_train_synthetic_leave_one_out_v1'
        ),
        'checkpoint_path': checkpoint_path,
        'dataset_provenance': profile_provenance(profile, data_path),
        'pseudoclass_dir': pseudoclass_dir,
        'num_records': len(records),
        'topk_ratios': topk_ratios,
        'corruption': {
            'grid': parsed_args.corruption_grid,
            'min_ratio': parsed_args.corruption_min_ratio,
            'max_ratio': parsed_args.corruption_max_ratio,
            'strength': parsed_args.corruption_strength,
            'seed': parsed_args.corruption_seed,
        },
        'reference_preserving': bool(parsed_args.reference_preserving),
    }
    with open(os.path.join(output_dir, 'calibration_summary.json'), 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    print(json.dumps({'num_records': len(records), 'output_dir': output_dir}, indent=2))


def select_global(parsed_args):
    input_dirs = [os.path.realpath(value) for value in parsed_args.input_dirs.split(',') if value.strip()]
    if len(input_dirs) < 2:
        raise ValueError('global selection requires at least two calibration directories')
    output_dir = os.path.realpath(parsed_args.output_dir)
    if os.path.exists(output_dir):
        raise FileExistsError('output_dir must not already exist: {}'.format(output_dir))
    frames = []
    sources = []
    for input_dir in input_dirs:
        with open(os.path.join(input_dir, 'calibration_summary.json'), encoding='utf-8') as file:
            summary = json.load(file)
        if summary.get('protocol') not in {
            'label_free_train_synthetic_leave_one_out_v1',
            'label_free_train_synthetic_reference_preserving_v2',
        }:
            raise ValueError('unsupported calibration protocol: {}'.format(input_dir))
        frame = pd.read_csv(os.path.join(input_dir, 'calibration_records.csv'))
        frame['calibration_source'] = os.path.basename(input_dir)
        frames.append(frame)
        sources.append(summary)
    records_df = pd.concat(frames, ignore_index=True)
    protocols = {summary.get('protocol') for summary in sources}
    if len(protocols) != 1:
        raise ValueError('global selection cannot mix calibration protocols')
    if 'topk_ratio' not in records_df.columns:
        raise ValueError('calibration records must contain topk_ratio')
    candidates = []
    lambda_candidates = _float_list(parsed_args.lambda_candidates, 'lambda_candidates', minimum=0.0)
    route_quantiles = _float_list(
        parsed_args.route_margin_quantiles,
        'route_margin_quantiles',
        minimum=0.0,
        maximum=1.0,
    )
    for _, topk_frame in records_df.groupby('topk_ratio', sort=True):
        result = select_tailguard_memory_calibration(
            topk_frame.to_dict(orient='records'),
            lambda_candidates=lambda_candidates,
            route_margin_quantiles=route_quantiles,
            max_clean_fusion_penalty=parsed_args.max_clean_fusion_penalty,
        )
        candidates.extend(result['candidates'])
    eligible = [row for row in candidates if row['eligible']] or candidates
    selected = dict(max(
        eligible,
        key=lambda row: (
            row['synthetic_auroc'],
            row['synthetic_localization'],
            -row['mean_clean_fusion_penalty'],
            row['corrupted_memory_applied_ratio'],
            -row['lambda'],
            -row['topk_ratio'],
        ),
    ))
    os.makedirs(output_dir)
    records_df.to_csv(os.path.join(output_dir, 'calibration_records.csv'), index=False)
    pd.DataFrame(candidates).to_csv(os.path.join(output_dir, 'calibration_candidates.csv'), index=False)
    payload = {
        'schema_version': 1,
        'protocol': '{}_global'.format(next(iter(protocols))),
        'selection_scope': 'all_sources_one_shared_configuration',
        'sources': sources,
        'num_records': len(records_df),
        'selected': selected,
    }
    with open(os.path.join(output_dir, 'calibration_summary.json'), 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    print(json.dumps(selected, indent=2, ensure_ascii=False))


def merge_global(parsed_args):
    input_dirs = [os.path.realpath(value) for value in parsed_args.input_dirs.split(',') if value.strip()]
    if len(input_dirs) < 2:
        raise ValueError('global merge requires at least two calibration directories')
    output_dir = os.path.realpath(parsed_args.output_dir)
    if os.path.exists(output_dir):
        raise FileExistsError('output_dir must not already exist: {}'.format(output_dir))
    frames = []
    sources = []
    for input_dir in input_dirs:
        with open(os.path.join(input_dir, 'calibration_summary.json'), encoding='utf-8') as file:
            summary = json.load(file)
        if summary.get('protocol') not in {
            'label_free_train_synthetic_leave_one_out_v1',
            'label_free_train_synthetic_reference_preserving_v2',
        }:
            raise ValueError('unsupported calibration protocol: {}'.format(input_dir))
        frame = pd.read_csv(os.path.join(input_dir, 'calibration_records.csv'))
        frame['calibration_source'] = os.path.basename(input_dir)
        frames.append(frame)
        sources.append(summary)
    os.makedirs(output_dir)
    protocols = {summary.get('protocol') for summary in sources}
    if len(protocols) != 1:
        raise ValueError('global merge cannot mix calibration protocols')
    records_df = pd.concat(frames, ignore_index=True)
    records_df.to_csv(os.path.join(output_dir, 'calibration_records.csv'), index=False)
    payload = {
        'schema_version': 1,
        'protocol': '{}_merged'.format(next(iter(protocols))),
        'selection_scope': 'all_sources_one_shared_configuration',
        'sources': sources,
        'num_records': len(records_df),
    }
    with open(os.path.join(output_dir, 'calibration_summary.json'), 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    print(json.dumps({'num_records': len(records_df), 'output_dir': output_dir}, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    collect_parser = subparsers.add_parser('collect')
    collect_parser.add_argument('--checkpoint_path', required=True)
    collect_parser.add_argument('--pseudoclass_dir', required=True)
    collect_parser.add_argument('--output_dir', required=True)
    collect_parser.add_argument('--dataset_profile', choices=dataset_profile_names(), default=None)
    collect_parser.add_argument('--data_path', default=None)
    collect_parser.add_argument('--gpu', type=int, default=0)
    collect_parser.add_argument('--batch_size', type=int, default=None)
    collect_parser.add_argument('--num_workers', type=int, default=None)
    collect_parser.add_argument('--tg_memory_topk_ratios', default='0.01,0.03,0.05,0.10')
    collect_parser.add_argument('--tg_memory_min_class_members', type=int, default=1)
    collect_parser.add_argument('--corruption_grid', type=int, default=8)
    collect_parser.add_argument('--corruption_min_ratio', type=float, default=0.2)
    collect_parser.add_argument('--corruption_max_ratio', type=float, default=0.5)
    collect_parser.add_argument('--corruption_strength', type=float, default=1.0)
    collect_parser.add_argument('--corruption_seed', type=int, default=2027)
    collect_parser.add_argument(
        '--reference_preserving',
        action='store_true',
        help='keep the clean training reference in memory when scoring its synthetic query',
    )
    select_parser = subparsers.add_parser('select')
    select_parser.add_argument('--input_dirs', required=True)
    select_parser.add_argument('--output_dir', required=True)
    select_parser.add_argument('--lambda_candidates', default='0,0.25,0.5,0.75,1.0')
    select_parser.add_argument('--route_margin_quantiles', default='0.0,0.1,0.25,0.5,0.75,0.9')
    select_parser.add_argument('--max_clean_fusion_penalty', type=float, default=None)
    merge_parser = subparsers.add_parser('merge')
    merge_parser.add_argument('--input_dirs', required=True)
    merge_parser.add_argument('--output_dir', required=True)
    return parser


if __name__ == '__main__':
    parsed = build_parser().parse_args()
    if parsed.command == 'collect':
        collect_calibration(parsed)
    elif parsed.command == 'select':
        select_global(parsed)
    else:
        merge_global(parsed)
