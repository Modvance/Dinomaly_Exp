import argparse
import os
import time

import numpy as np
import torch

from dinomaly_train_base import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_TOTAL_ITERS,
    build_datasets,
    build_model,
    build_optimizer_and_scheduler,
    build_phase6_train_dataloader,
    build_train_eval_dataloader,
    evaluate_model,
    get_logger,
    save_train_checkpoint,
    score_trainset,
    setup_seed,
    train_step,
)
from phase6_runner_utils import build_longtail_sampling_plan, save_longtail_artifacts

warnings = __import__('warnings')
warnings.filterwarnings('ignore')


def prepare_true_class_longtail_plan(train_data_list, item_list, total_train_samples):
    model, _ = build_model(device)
    train_eval_dataloader = build_train_eval_dataloader(
        train_data_list,
        data_root=args.data_path,
        batch_size=args.diag_batch_size,
        num_workers=args.diag_num_workers,
        item_list=item_list,
        contaminated_paths=None,
    )
    scored_df = score_trainset(
        model,
        train_eval_dataloader,
        device,
        max_ratio=args.diag_max_ratio,
        resize_mask=args.diag_resize_mask,
    )
    scored_df['group_id'] = scored_df['class_id'].astype(int)
    scored_df['pseudo_group_id'] = scored_df['class_id'].astype(int)
    sampling_plan = build_longtail_sampling_plan(
        scored_df,
        args,
        original_num_samples=total_train_samples,
        prune_group_counts=None,
    )
    return sampling_plan


def train(item_list):
    setup_seed(1)

    total_iters = DEFAULT_TOTAL_ITERS
    batch_size = DEFAULT_BATCH_SIZE
    num_workers = DEFAULT_NUM_WORKERS
    train_start_time = time.time()
    final_eval_summary = None

    train_data_list, test_data_list = build_datasets(args.data_path, item_list)
    total_train_samples = int(sum(len(dataset) for dataset in train_data_list))
    print_fn('phase6 true-class pretrain scoring start: total_train_samples={}, lt_group_col={}'.format(
        total_train_samples,
        args.lt_group_col,
    ))
    lt_sampling_plan = prepare_true_class_longtail_plan(train_data_list, item_list, total_train_samples)
    extra_summary = {
        'phase6_group_source': 'true_class',
        'phase6_true_class_mode': True,
        'phase6_scoring_mode': 'pretrain_score_only',
        'phase6_gbps_disabled': True,
    }
    save_longtail_artifacts(args.diag_save_dir, lt_sampling_plan, extra_summary=extra_summary)
    print_fn('phase6 true-class sampler prepared: groups={}, epoch_num_samples={}, alpha={:.3f}'.format(
        lt_sampling_plan['summary']['lt_total_groups'],
        lt_sampling_plan['summary']['lt_epoch_num_samples_effective'],
        lt_sampling_plan['summary']['lt_sampler_alpha'],
    ))

    model, trainable = build_model(device)
    optimizer, lr_scheduler = build_optimizer_and_scheduler(trainable, total_iters=total_iters)
    train_data, train_dataloader = build_phase6_train_dataloader(
        train_data_list,
        data_root=args.data_path,
        item_list=item_list,
        phase6_plan=lt_sampling_plan,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=None if not args.lt_sampler_enable else lt_sampling_plan['sampler_weights'],
        epoch_num_samples=None if not args.lt_sampler_enable else lt_sampling_plan['epoch_num_samples'],
    )
    print_fn('phase6 true-class training image number:{}'.format(len(train_data)))

    it = 0
    for _ in range(int(np.ceil(total_iters / len(train_dataloader)))):
        loss_list = []
        for batch in train_dataloader:
            if len(batch) == 3:
                images, _, meta = batch
                sample_weight = None
                if args.lt_group_balanced_loss and isinstance(meta, dict) and 'loss_weight' in meta:
                    sample_weight = meta['loss_weight']
            else:
                images, _ = batch
                sample_weight = None

            loss_value = train_step(
                model,
                trainable,
                optimizer,
                lr_scheduler,
                images,
                current_zero_based_iter=it,
                sample_weight=sample_weight,
            )
            loss_list.append(loss_value)

            current_iter = it + 1
            if current_iter % args.eval_interval == 0:
                final_eval_summary = evaluate_model(
                    model,
                    test_data_list,
                    item_list,
                    device,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    max_ratio=0.01,
                    resize_mask=256,
                    print_fn=print_fn,
                )
                model.train()

            it += 1
            if it == total_iters:
                break
        print_fn('phase6 iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
        if it == total_iters:
            break

    if final_eval_summary is None:
        final_eval_summary = evaluate_model(
            model,
            test_data_list,
            item_list,
            device,
            batch_size=batch_size,
            num_workers=num_workers,
            max_ratio=0.01,
            resize_mask=256,
            print_fn=print_fn,
        )
        model.train()

    checkpoint_path = save_train_checkpoint(
        args.save_dir,
        args.save_name,
        model,
        iteration=it,
        args=args,
        final_eval_summary=final_eval_summary,
    )

    total_time = time.time() - train_start_time
    time_per_iter = total_time / total_iters
    iters_per_sec = total_iters / total_time
    samples_per_sec = batch_size / time_per_iter

    print_fn('Phase6 true-class training finished. Total time: {:.2f}s ({:.2f} min), {:.4f} s/iter'.format(
        total_time,
        total_time / 60.0,
        time_per_iter,
    ))
    print_fn('phase6 true-class final checkpoint saved: {}'.format(checkpoint_path))
    if final_eval_summary is not None:
        print_fn('I-AUROC      {:.2f}'.format(final_eval_summary['I-AUROC']))
        print_fn('I-AP         {:.2f}'.format(final_eval_summary['I-AP']))
        print_fn('I-F1         {:.2f}'.format(final_eval_summary['I-F1']))
        print_fn('P-AUROC      {:.2f}'.format(final_eval_summary['P-AUROC']))
        print_fn('P-AP         {:.2f}'.format(final_eval_summary['P-AP']))
        print_fn('P-F1         {:.2f}'.format(final_eval_summary['P-F1']))
        print_fn('P-AUPRO      {:.2f}'.format(final_eval_summary['P-AUPRO']))
    print_fn('Training efficiency:')
    print_fn('  total_time_s      {:.2f}'.format(total_time))
    print_fn('  total_iters       {}'.format(total_iters))
    print_fn('  batch_size        {}'.format(batch_size))
    print_fn('  time_per_iter_s   {:.4f}'.format(time_per_iter))
    print_fn('  iters_per_sec     {:.4f}'.format(iters_per_sec))
    print_fn('  samples_per_sec   {:.2f}'.format(samples_per_sec))


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    parser = argparse.ArgumentParser(description='Standalone true-class phase6 runner for MVTec')
    parser.add_argument('--data_path', type=str, default='../mvtec_anomaly_detection')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='vitill_mvtec_phase6_true_class')
    parser.add_argument('--diag_save_dir', type=str, default='./phase6_diag')
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--eval_interval', type=int, default=5000)
    parser.add_argument('--diag_batch_size', type=int, default=16)
    parser.add_argument('--diag_num_workers', type=int, default=4)
    parser.add_argument('--diag_max_ratio', type=float, default=0.01)
    parser.add_argument('--diag_resize_mask', type=int, default=256)
    parser.add_argument('--lt_enable', action='store_true', default=True)
    parser.add_argument('--lt_group_col', type=str, default='class_id')
    parser.add_argument('--lt_sampler_enable', action='store_true')
    parser.add_argument('--lt_sampler_alpha', type=float, default=0.5)
    parser.add_argument('--lt_sampler_clip_min', type=float, default=0.25)
    parser.add_argument('--lt_sampler_clip_max', type=float, default=4.0)
    parser.add_argument('--lt_epoch_num_samples', type=str, default='kept', choices=['kept', 'original'])
    parser.add_argument('--lt_group_balanced_loss', action='store_true')
    parser.add_argument('--lt_tail_aug_enable', action='store_true')
    parser.add_argument('--lt_tail_rho_thr', type=float, default=0.3)
    parser.add_argument('--lt_tail_aug_p_noisy_thr', type=float, default=0.2)
    parser.add_argument('--lt_tail_aug_k', type=int, default=4)
    parser.add_argument('--lt_tail_aug_max_group_ratio', type=float, default=0.5)
    parser.add_argument('--lt_tail_aug_score_eta', type=float, default=0.5)
    parser.add_argument('--lt_aug_weight_scale', type=float, default=0.5)
    parser.add_argument('--lt_tail_aug_use_warmup_score', action='store_true')
    parser.add_argument('--lt_tail_aug_manifest_path', type=str, default=None)
    args = parser.parse_args()

    item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
                 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    if not os.path.isabs(args.diag_save_dir):
        args.diag_save_dir = os.path.join(args.save_dir, args.save_name, args.diag_save_dir)

    device = 'cuda:{}'.format(args.gpus) if torch.cuda.is_available() else 'cpu'
    print_fn(device)
    train(item_list)
