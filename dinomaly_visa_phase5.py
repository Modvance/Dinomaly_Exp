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
    build_phase5_train_dataloader,
    build_train_eval_dataloader,
    evaluate_model,
    get_logger,
    setup_seed,
    train_phase5_step,
)
from phase5_reliability import prepare_phase5_reliability
from phase5_runner_utils import (
    build_phase5_analysis_artifacts,
    build_phase5_weight_summary_from_bank,
    resolve_phase5_manifest_info,
    save_phase5_artifacts,
    save_phase5_bank,
    save_phase5_checkpoint,
    save_phase5_eval_outputs,
)

warnings = __import__('warnings')
warnings.filterwarnings('ignore')


def train(item_list):
    setup_seed(1)

    total_iters = DEFAULT_TOTAL_ITERS
    batch_size = DEFAULT_BATCH_SIZE
    num_workers = DEFAULT_NUM_WORKERS
    train_start_time = time.time()
    final_eval_summary = None
    eval_history_rows = []

    manifest_info = resolve_phase5_manifest_info(args.data_path, requested_manifest_path=args.diag_manifest_path)
    train_data_list, test_data_list = build_datasets(args.data_path, item_list)
    train_eval_dataloader = build_train_eval_dataloader(
        train_data_list,
        data_root=args.data_path,
        batch_size=args.diag_batch_size,
        num_workers=args.diag_num_workers,
        item_list=item_list,
        contaminated_paths=manifest_info['contaminated_paths'],
    )

    model, _ = build_model(device)
    reliability_result = prepare_phase5_reliability(
        model,
        train_eval_dataloader,
        device,
        args.diag_save_dir,
        args,
    )
    reliability_bank = reliability_result['reliability_bank']
    reliability_summary_df = reliability_result['summary_df']
    reliability_global_summary = dict(reliability_result['global_summary'])
    reliability_global_summary.update(manifest_info)
    reliability_global_summary['run_dir'] = args.diag_save_dir
    analysis_outputs = build_phase5_analysis_artifacts(reliability_summary_df, reliability_global_summary)
    reliability_global_summary = dict(analysis_outputs['summary'])
    save_phase5_artifacts(reliability_result['reliability_dir'], reliability_summary_df, reliability_global_summary, analysis_outputs=analysis_outputs)
    if args.phase5_reliability_cache_path is None:
        bank_path = reliability_result['cache_path']
    else:
        bank_path = save_phase5_bank(reliability_result['reliability_dir'], reliability_bank, args.phase5_reliability_cache_path)
    print_fn('phase5 groups prepared: selected_k={}, loaded_from_cache={}'.format(
        reliability_result['group_info'].get('selected_k'),
        reliability_result['group_info'].get('loaded_from_cache'),
    ))
    print_fn('phase5 reliability bank saved: {}, loaded_from_cache={}'.format(
        bank_path,
        reliability_global_summary.get('loaded_from_cache'),
    ))

    phase5_weight_summary_df = build_phase5_weight_summary_from_bank(reliability_bank)
    skip_phase5_weighting = bool((phase5_weight_summary_df['suppress_ratio'] <= 0).all()) if len(phase5_weight_summary_df) > 0 else True

    model, trainable = build_model(device)
    optimizer, lr_scheduler = build_optimizer_and_scheduler(trainable, total_iters=total_iters)
    train_data, train_dataloader = build_phase5_train_dataloader(
        train_data_list,
        data_root=args.data_path,
        item_list=item_list,
        reliability_bank=reliability_bank,
        patch_grid_size=args.phase5_patch_grid_size,
        batch_size=batch_size,
        num_workers=num_workers,
        skip_phase5_weighting=skip_phase5_weighting,
    )
    print_fn('phase5 restart retrain image number:{}'.format(len(train_data)))

    it = 0
    for _ in range(int(np.ceil(total_iters / len(train_dataloader)))):
        loss_list = []
        for images, _, meta in train_dataloader:
            loss_value = train_phase5_step(
                model,
                trainable,
                optimizer,
                lr_scheduler,
                images,
                meta['w_img'],
                meta['w_patch'],
                current_zero_based_iter=it,
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
                eval_row = {'iteration': int(current_iter)}
                eval_row.update(final_eval_summary)
                eval_history_rows.append(eval_row)
                model.train()

            it += 1
            if it == total_iters:
                break
        print_fn('phase5 iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
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
        eval_row = {'iteration': int(total_iters)}
        eval_row.update(final_eval_summary)
        eval_history_rows.append(eval_row)
        model.train()

    final_eval_payload = {
        'iteration': int(it),
        'run_dir': args.diag_save_dir,
        'save_name': args.save_name,
    }
    final_eval_payload.update(final_eval_summary)
    final_eval_summary_path = save_phase5_eval_outputs(args.diag_save_dir, final_eval_payload, eval_history_rows=eval_history_rows)
    checkpoint_path = save_phase5_checkpoint(
        args.diag_save_dir,
        model,
        iteration=it,
        args=args,
        extra_state={'final_eval_summary': final_eval_payload},
    )

    total_time = time.time() - train_start_time
    print_fn('Phase5 training finished. Total time: {:.2f}s ({:.2f} min)'.format(total_time, total_time / 60.0))
    print_fn('phase5 final eval saved: {}'.format(final_eval_summary_path))
    print_fn('phase5 final checkpoint saved: {}'.format(checkpoint_path))
    if final_eval_summary is not None:
        print_fn('I-AUROC      {:.2f}'.format(final_eval_summary['I-AUROC']))
        print_fn('I-AP         {:.2f}'.format(final_eval_summary['I-AP']))
        print_fn('I-F1         {:.2f}'.format(final_eval_summary['I-F1']))
        print_fn('P-AUROC      {:.2f}'.format(final_eval_summary['P-AUROC']))
        print_fn('P-AP         {:.2f}'.format(final_eval_summary['P-AP']))
        print_fn('P-F1         {:.2f}'.format(final_eval_summary['P-F1']))
        print_fn('P-AUPRO      {:.2f}'.format(final_eval_summary['P-AUPRO']))


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    parser = argparse.ArgumentParser(description='Standalone phase5 runner for VisA')
    parser.add_argument('--data_path', type=str, default='../VisA_pytorch/1cls')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='vitill_visa_phase5')
    parser.add_argument('--diag_save_dir', type=str, default='./phase5_diag')
    parser.add_argument('--diag_manifest_path', type=str, default=None)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--eval_interval', type=int, default=5000)
    parser.add_argument('--diag_batch_size', type=int, default=16)
    parser.add_argument('--diag_num_workers', type=int, default=4)
    parser.add_argument('--gbps_grouping_method', type=str, default='kmeans')
    parser.add_argument('--gbps_num_groups', type=int, default=None)
    parser.add_argument('--gbps_auto_k', action='store_true')
    parser.add_argument('--gbps_k_candidates', type=str, default='8,10,12,15,20')
    parser.add_argument('--gbps_pca_dim', type=int, default=64)
    parser.add_argument('--gbps_embedding_source', type=str, default='encoder')
    parser.add_argument('--gbps_group_cache_path', type=str, default=None)
    parser.add_argument('--phase5_knn_k', type=int, default=5)
    parser.add_argument('--phase5_memory_size', type=int, default=1024)
    parser.add_argument('--phase5_memory_build_mode', type=str, default='kmeans')
    parser.add_argument('--phase5_top_wimg_ratio', type=float, default=0.5)
    parser.add_argument('--phase5_reliability_cache_path', type=str, default=None)
    parser.add_argument('--phase5_force_recompute', action='store_true')
    parser.add_argument('--phase5_min_group_size', type=int, default=3)
    parser.add_argument('--phase5_eps', type=float, default=1e-6)
    parser.add_argument('--phase5_patch_grid_size', type=str, default='28,28')
    parser.add_argument('--phase5_loss_norm', type=str, default='per_image', choices=['per_image'])
    parser.add_argument('--phase5_train_mode', type=str, default='restart', choices=['restart'])
    args = parser.parse_args()

    item_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    if not os.path.isabs(args.diag_save_dir):
        args.diag_save_dir = os.path.join(args.save_dir, args.save_name, args.diag_save_dir)
    if args.phase5_reliability_cache_path is None:
        args.phase5_reliability_cache_path = os.path.join(args.diag_save_dir, 'phase5_reliability', 'reliability_bank.pt')

    device = 'cuda:{}'.format(args.gpus) if torch.cuda.is_available() else 'cpu'
    print_fn(device)
    train(item_list)
