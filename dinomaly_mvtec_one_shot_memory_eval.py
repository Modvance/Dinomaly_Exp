import argparse
import json
import os

import torch

from dinomaly_train_base import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    build_datasets,
    build_train_eval_dataloader,
    get_class_train_counts,
    get_logger,
    load_train_checkpoint_metadata,
    load_model_from_train_checkpoint,
)
from one_shot_memory import (
    MVTec_ITEM_LIST,
    build_class_memory_bank,
    run_memory_augmented_evaluation,
    save_memory_artifacts,
)

warnings = __import__('warnings')
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Standalone MVTec one-shot class-wise patch memory eval')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='../mvtec_anomaly_detection')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='vitill_mvtec_one_shot_memory_eval')
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument('--max_ratio', type=float, default=0.01)
    parser.add_argument('--resize_mask', type=int, default=256)
    parser.add_argument('--mem_enable', action='store_true', default=False)
    parser.add_argument('--mem_apply_mode', type=str, default='class_size_le', choices=['all', 'one_shot', 'class_size_le'])
    parser.add_argument('--mem_class_size_thr', type=int, default=8)
    parser.add_argument('--mem_topk_ratio', type=float, default=0.05)
    parser.add_argument('--mem_fusion_lambda', type=float, default=1.0)
    parser.add_argument('--mem_score_mode', type=str, default='fused', choices=['recon', 'memory', 'fused'])
    parser.add_argument('--mem_chunk_size', type=int, default=4096)
    parser.add_argument('--mem_artifact_dir', type=str, default='one_shot_memory')
    parser.add_argument('--mem_load_path', type=str, default=None)
    parser.add_argument('--mem_max_patches_per_class', type=int, default=20000)
    args = parser.parse_args()

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:{}'.format(args.gpus) if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    model, _, checkpoint_metadata = load_model_from_train_checkpoint(args.checkpoint_path, device)
    train_data_list, test_data_list = build_datasets(
        args.data_path,
        MVTec_ITEM_LIST,
        image_size=checkpoint_metadata['image_size'],
        crop_size=checkpoint_metadata['crop_size'],
    )
    class_train_counts = get_class_train_counts(train_data_list, MVTec_ITEM_LIST)
    train_eval_dataloader = build_train_eval_dataloader(
        train_data_list,
        data_root=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        item_list=MVTec_ITEM_LIST,
        contaminated_paths=None,
    )

    if args.mem_load_path:
        memory_bank = torch.load(args.mem_load_path, map_location='cpu')
        print_fn('loaded memory bank: {}'.format(args.mem_load_path))
    else:
        memory_bank = build_class_memory_bank(
            model,
            train_eval_dataloader,
            device,
            class_train_counts,
            args,
            item_list=MVTec_ITEM_LIST,
        )
        print_fn('built memory bank for {} classes'.format(len(memory_bank)))

    score_df, per_class_metrics, summary = run_memory_augmented_evaluation(
        model,
        test_data_list,
        MVTec_ITEM_LIST,
        memory_bank,
        args,
        device,
    )

    output_dir = os.path.join(args.save_dir, args.save_name, args.mem_artifact_dir)
    metadata = {
        'checkpoint_path': args.checkpoint_path,
        'checkpoint_iteration': checkpoint_metadata['iteration'],
        'encoder_name': checkpoint_metadata['encoder_name'],
        'image_size': checkpoint_metadata['image_size'],
        'crop_size': checkpoint_metadata['crop_size'],
        'max_ratio': args.max_ratio,
        'resize_mask': args.resize_mask,
        'mem_apply_mode': args.mem_apply_mode,
        'mem_class_size_thr': args.mem_class_size_thr,
        'mem_topk_ratio': args.mem_topk_ratio,
        'mem_fusion_lambda': args.mem_fusion_lambda,
        'mem_score_mode': args.mem_score_mode,
        'class_train_counts': class_train_counts,
        'final_eval_summary': checkpoint_metadata['final_eval_summary'],
    }
    saved = save_memory_artifacts(
        output_dir,
        memory_bank,
        score_df,
        per_class_metrics,
        summary,
        metadata,
    )

    print_fn('saved memory artifacts to {}'.format(output_dir))
    print_fn(json.dumps(summary, indent=2, ensure_ascii=False))
    print_fn(json.dumps(saved, indent=2, ensure_ascii=False))
