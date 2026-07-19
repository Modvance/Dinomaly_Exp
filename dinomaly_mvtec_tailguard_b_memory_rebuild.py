import argparse
import json
import math
import os
from types import SimpleNamespace

import pandas as pd
import torch

from dinomaly_train_base import (
    build_datasets,
    load_model_from_train_checkpoint,
    rebuild_pruned_train_loaders,
)
from one_shot_memory import MVTec_ITEM_LIST
from tailguard_b_artifacts import save_tailguard_b_memory_artifacts
from tailguard_b_gbps import _build_retained_index_map
from tailguard_b_memory import (
    build_tailguard_b_pseudoclass_memory_system,
    run_pseudoclass_memory_evaluation,
)


def _checkpoint_value(metadata, name, default):
    checkpoint_args = metadata.get('args') or {}
    if isinstance(checkpoint_args, dict):
        return checkpoint_args.get(name, default)
    return getattr(checkpoint_args, name, default)


def _resolve(value, metadata, name, default):
    resolved = _checkpoint_value(metadata, name, default) if value is None else value
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


def _retained_index_map(members_df, train_data_list):
    required_columns = {'class_id', 'base_idx'}
    missing_columns = required_columns.difference(members_df.columns)
    if missing_columns:
        raise ValueError(
            'pseudo-class members are missing stable identities: {}. Regenerate them with '
            'dinomaly_mvtec_tailguard_b_attachment_replay.py before rebuilding memory.'.format(
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
    data_path = os.path.realpath(parsed_args.data_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError('final model checkpoint does not exist: {}'.format(checkpoint_path))
    if not os.path.isdir(pseudoclass_dir):
        raise FileNotFoundError('pseudo-class directory does not exist: {}'.format(pseudoclass_dir))
    if not os.path.isdir(data_path):
        raise FileNotFoundError('dataset directory does not exist: {}'.format(data_path))
    if os.path.exists(output_dir):
        raise FileExistsError('output_dir must not already exist: {}'.format(output_dir))

    members_df = pd.read_csv(os.path.join(pseudoclass_dir, 'pseudo_class_members.csv'))
    classes_df = pd.read_csv(os.path.join(pseudoclass_dir, 'pseudo_classes.csv'))
    gpu = _integer_value(parsed_args.gpu, 'gpu', 0)
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
    model, _, checkpoint_metadata = load_model_from_train_checkpoint(checkpoint_path, device)
    checkpoint_provenance = {
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
        tgb_memory_fusion_lambda=float(_resolve(
            parsed_args.tgb_memory_fusion_lambda,
            checkpoint_metadata,
            'tgb_memory_fusion_lambda',
            1.0,
        )),
        tgb_memory_topk_ratio=_ratio_value(
            _resolve(parsed_args.tgb_memory_topk_ratio, checkpoint_metadata, 'tgb_memory_topk_ratio', 0.05),
            'tgb_memory_topk_ratio',
        ),
        tgb_mem_chunk_size=_integer_value(
            _resolve(parsed_args.tgb_mem_chunk_size, checkpoint_metadata, 'tgb_mem_chunk_size', 4096),
            'tgb_mem_chunk_size',
            1,
        ),
        tgb_mem_max_patches_per_class=_integer_value(
            _resolve(
                parsed_args.tgb_mem_max_patches_per_class,
                checkpoint_metadata,
                'tgb_mem_max_patches_per_class',
                20000,
            ),
            'tgb_mem_max_patches_per_class',
            0,
        ),
    )
    if not math.isfinite(memory_args.tgb_memory_fusion_lambda) or memory_args.tgb_memory_fusion_lambda < 0.0:
        raise ValueError('tgb_memory_fusion_lambda must be finite and non-negative')
    del checkpoint_metadata
    train_data_list, test_data_list = build_datasets(
        data_path,
        MVTec_ITEM_LIST,
        image_size=checkpoint_provenance['image_size'],
        crop_size=checkpoint_provenance['crop_size'],
    )
    loaders = rebuild_pruned_train_loaders(
        train_data_list,
        _retained_index_map(members_df, train_data_list),
        data_root=data_path,
        item_list=MVTec_ITEM_LIST,
        batch_size=batch_size,
        num_workers=num_workers,
        diag_batch_size=diag_batch_size,
        diag_num_workers=diag_num_workers,
    )
    memory_system = build_tailguard_b_pseudoclass_memory_system(
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
        MVTec_ITEM_LIST,
        memory_system,
        memory_args,
        device,
    )
    os.makedirs(output_dir, exist_ok=True)
    artifacts = save_tailguard_b_memory_artifacts(
        output_dir,
        memory_system,
        score_df,
        per_class_metrics,
        summary,
        {
            'checkpoint_provenance': checkpoint_provenance,
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
            'tgb_memory_fusion_lambda': memory_args.tgb_memory_fusion_lambda,
            'tgb_memory_topk_ratio': memory_args.tgb_memory_topk_ratio,
            'tgb_mem_chunk_size': memory_args.tgb_mem_chunk_size,
            'tgb_mem_max_patches_per_class': memory_args.tgb_mem_max_patches_per_class,
        },
        metrics_by_mode=metrics_by_mode,
    )
    result = {
        'checkpoint_provenance': checkpoint_provenance,
        'pseudoclass_dir': pseudoclass_dir,
        'data_path': data_path,
        'memory_artifacts': artifacts,
        'memory_eval_summary': summary,
    }
    with open(os.path.join(output_dir, 'memory_rebuild_summary.json'), 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build TailGuard-B final memory from a saved final model without retraining')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--pseudoclass_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='../mvtec_anomaly_detection')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--diag_batch_size', type=int, default=None)
    parser.add_argument('--diag_num_workers', type=int, default=None)
    parser.add_argument('--diag_max_ratio', type=float, default=None)
    parser.add_argument('--diag_resize_mask', type=int, default=None)
    parser.add_argument('--tgb_memory_fusion_lambda', type=float, default=None)
    parser.add_argument('--tgb_memory_topk_ratio', type=float, default=None)
    parser.add_argument('--tgb_mem_chunk_size', type=int, default=None)
    parser.add_argument('--tgb_mem_max_patches_per_class', type=int, default=None)
    rebuild_memory(parser.parse_args())
