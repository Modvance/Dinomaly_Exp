import argparse
import json
import os

import torch

from dinomaly_train_base import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_CROP_SIZE,
    DEFAULT_ENCODER_NAME,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_TARGET_LAYERS,
    DEFAULT_FUSE_LAYER_ENCODER,
    DEFAULT_FUSE_LAYER_DECODER,
    build_datasets,
    build_model,
    build_train_eval_dataloader,
    get_logger,
    load_model_from_train_checkpoint,
)
from feature_grouping import extract_image_embeddings
from tail_sampler_analysis import run_tail_sampler_analysis, save_tail_sampler_artifacts

warnings = __import__('warnings')
warnings.filterwarnings('ignore')


MVTec_ITEM_LIST = [
    'carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
    'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper',
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Standalone TailSampler analysis for Dinomaly MVTec train embeddings')
    parser.add_argument('--data_path', type=str, default='../mvtec_anomaly_detection')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='vitill_mvtec_tailsampler')
    parser.add_argument('--diag_save_dir', type=str, default='tailsampler_diag')
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--diag_batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--diag_num_workers', type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--image_size', type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument('--crop_size', type=int, default=DEFAULT_CROP_SIZE)
    parser.add_argument('--encoder_name', type=str, default=DEFAULT_ENCODER_NAME)
    parser.add_argument('--tailsampler_type', type=str, default='adaptive_trim_mode', choices=['tail', 'adaptive', 'adaptive_trim_mode'])
    parser.add_argument('--tailsampler_th_type', type=str, default=None)
    parser.add_argument('--tailsampler_vote_type', type=str, default=None)
    parser.add_argument('--tailsampler_percentile', type=float, default=0.15)
    parser.add_argument('--tailsampler_gt_mode', type=str, default='true_class', choices=['true_class', 'dataset_rule'])
    parser.add_argument('--tailsampler_tail_count_thr', type=int, default=8)
    parser.add_argument('--tailsampler_dataset_name', type=str, default=None)
    parser.add_argument('--tailsampler_embedding_source', type=str, default='encoder')
    parser.add_argument('--save_sampler_details', action='store_true')
    args = parser.parse_args()

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    device = 'cuda:{}'.format(args.gpus) if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    checkpoint_metadata = None
    if args.checkpoint_path:
        model, _, checkpoint_metadata = load_model_from_train_checkpoint(args.checkpoint_path, device)
        image_size = checkpoint_metadata['image_size']
        crop_size = checkpoint_metadata['crop_size']
        print_fn('loaded checkpoint: {}'.format(args.checkpoint_path))
    else:
        model, _ = build_model(
            device,
            encoder_name=args.encoder_name,
            target_layers=DEFAULT_TARGET_LAYERS,
            fuse_layer_encoder=DEFAULT_FUSE_LAYER_ENCODER,
            fuse_layer_decoder=DEFAULT_FUSE_LAYER_DECODER,
        )
        image_size = args.image_size
        crop_size = args.crop_size
        print_fn('built model from encoder: {}'.format(args.encoder_name))

    train_data_list, _ = build_datasets(
        args.data_path,
        MVTec_ITEM_LIST,
        image_size=image_size,
        crop_size=crop_size,
    )
    train_eval_dataloader = build_train_eval_dataloader(
        train_data_list,
        data_root=args.data_path,
        batch_size=args.diag_batch_size,
        num_workers=args.diag_num_workers,
        item_list=MVTec_ITEM_LIST,
        contaminated_paths=None,
    )

    embeddings, metadata_df = extract_image_embeddings(
        model,
        train_eval_dataloader,
        device,
        embedding_source=args.tailsampler_embedding_source,
    )
    print_fn('collected train embeddings: num_samples={}, dim={}'.format(embeddings.shape[0], embeddings.shape[1] if embeddings.ndim == 2 else 0))

    if args.tailsampler_dataset_name is None:
        args.tailsampler_dataset_name = os.path.basename(os.path.normpath(args.data_path))

    summary_row, details_df = run_tail_sampler_analysis(embeddings, metadata_df, args)

    output_dir = args.diag_save_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(args.save_dir, args.save_name, output_dir)

    artifact_metadata = {
        'data_path': args.data_path,
        'dataset_name': args.tailsampler_dataset_name,
        'checkpoint_path': args.checkpoint_path,
        'encoder_name': checkpoint_metadata['encoder_name'] if checkpoint_metadata is not None else args.encoder_name,
        'image_size': int(image_size),
        'crop_size': int(crop_size),
        'tailsampler_type': args.tailsampler_type,
        'tailsampler_th_type': args.tailsampler_th_type,
        'tailsampler_vote_type': args.tailsampler_vote_type,
        'tailsampler_percentile': float(args.tailsampler_percentile),
        'tailsampler_gt_mode': args.tailsampler_gt_mode,
        'tailsampler_tail_count_thr': int(args.tailsampler_tail_count_thr),
        'tailsampler_embedding_source': args.tailsampler_embedding_source,
    }

    saved = save_tail_sampler_artifacts(
        output_dir,
        summary_row,
        details_df,
        metadata=artifact_metadata,
        save_details=bool(args.save_sampler_details),
    )

    print_fn('saved TailSampler artifacts to {}'.format(output_dir))
    print_fn(json.dumps(summary_row, indent=2, ensure_ascii=False))
    print_fn(json.dumps(saved, indent=2, ensure_ascii=False))
