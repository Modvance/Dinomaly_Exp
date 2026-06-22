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
    build_patch_train_dataloader,
    build_train_dataloader,
    build_train_eval_dataloader,
    collect_train_anomaly_maps,
    evaluate_model,
    get_logger,
    setup_seed,
    should_run_check,
    train_patch_step,
    train_step,
)
from patch_diagnosis import (
    build_patch_diagnosis_table,
    build_patch_score_grid,
    build_patch_weight_bank,
    compute_patch_u_with_bootstrap,
    parse_patch_grid_size,
)
from patch_runner_utils import (
    build_patch_weight_summary_from_bank,
    evaluate_patch_ci_peak_trigger,
    resolve_best_patch_diagnosis,
    save_best_patch_checkpoint,
    save_patch_artifacts,
    save_patch_weight_artifacts,
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
    patch_grid_size = parse_patch_grid_size(args.patch_grid_size)

    train_data_list, test_data_list = build_datasets(args.data_path, item_list)
    train_data, train_dataloader = build_train_dataloader(train_data_list, batch_size=batch_size, num_workers=num_workers)
    train_eval_dataloader = build_train_eval_dataloader(
        train_data_list,
        data_root=args.data_path,
        batch_size=args.diag_batch_size,
        num_workers=args.diag_num_workers,
        item_list=item_list,
    )

    model, trainable = build_model(device)
    optimizer, lr_scheduler = build_optimizer_and_scheduler(trainable, total_iters=total_iters)

    print_fn('patch grid size:{}x{}'.format(patch_grid_size[0], patch_grid_size[1]))
    print_fn('train image number:{}'.format(len(train_data)))

    patch_best_U = None
    patch_best_SE = None
    patch_best_iter = None
    patch_best_active_ratio = None
    patch_best_check_count = None
    patch_check_count = 0
    patch_trigger_iter = None
    patch_selected_iter = None
    best_patch_payload = None
    best_patch_diagnosis_df = None
    best_patch_summary = None

    it = 0
    for _ in range(int(np.ceil(total_iters / len(train_dataloader)))):
        loss_list = []
        for images, _ in train_dataloader:
            loss_value = train_step(
                model,
                trainable,
                optimizer,
                lr_scheduler,
                images,
                current_zero_based_iter=it,
            )
            loss_list.append(loss_value)

            current_iter = it + 1
            if patch_trigger_iter is None and should_run_check(
                current_iter,
                total_iters,
                args.patch_check_start_ratio,
                args.patch_check_end_ratio,
                args.patch_check_interval,
            ):
                patch_check_count += 1
                iter_dir = os.path.join(args.diag_save_dir, 'iter_{:05d}'.format(current_iter))
                scored_df, anomaly_maps = collect_train_anomaly_maps(
                    model,
                    train_eval_dataloader,
                    device,
                    max_ratio=args.diag_max_ratio,
                    resize_mask=args.diag_resize_mask,
                    apply_smoothing=args.patch_use_smoothed_map,
                )
                patch_grids = build_patch_score_grid(anomaly_maps, patch_grid_size)
                diagnosis_df, diagnosis_payload = build_patch_diagnosis_table(
                    scored_df,
                    patch_grids,
                    args=args,
                )
                patch_result = compute_patch_u_with_bootstrap(
                    diagnosis_df,
                    B=args.patch_bootstrap_B,
                    args=args,
                )
                t_max = max(1, int(total_iters * args.patch_check_end_ratio))
                trigger_result = evaluate_patch_ci_peak_trigger(
                    U_t=patch_result['patch_U'],
                    SE_t=patch_result['patch_SE'],
                    active_ratio_t=patch_result['summary']['patch_active_ratio'],
                    best_U=patch_best_U,
                    best_SE=patch_best_SE,
                    best_iter=patch_best_iter,
                    best_active_ratio=patch_best_active_ratio,
                    best_check_count=patch_best_check_count,
                    check_count=patch_check_count,
                    current_iter=current_iter,
                    ci_z=args.patch_ci_z,
                    improve_eps=args.patch_improve_eps,
                    min_checks_before_trigger=args.patch_min_checks_before_trigger,
                    min_checks_after_best=args.patch_min_checks_after_best,
                    min_active_ratio=args.patch_min_active_ratio,
                    force_trigger=current_iter >= t_max,
                )
                patch_best_U = trigger_result['best_U']
                patch_best_SE = trigger_result['best_SE']
                patch_best_iter = trigger_result['best_iter']
                patch_best_active_ratio = trigger_result['best_active_ratio']
                patch_best_check_count = trigger_result['best_check_count']

                summary = dict(patch_result['summary'])
                summary.update(trigger_result['summary'])
                summary['iteration'] = int(current_iter)
                summary['patch_grid_size'] = [int(patch_grid_size[0]), int(patch_grid_size[1])]
                save_patch_artifacts(
                    iter_dir,
                    diagnosis_df=diagnosis_df,
                    bootstrap_df=patch_result['bootstrap_df'],
                    extra_summary=summary,
                )

                if trigger_result['summary']['patch_improved']:
                    best_patch_payload = diagnosis_payload
                    best_patch_diagnosis_df = diagnosis_df.copy()
                    best_patch_summary = dict(summary)
                    best_diagnosis_path = os.path.join(args.diag_save_dir, 'best_patch_diagnosis.csv')
                    best_patch_diagnosis_df.to_csv(best_diagnosis_path, index=False)
                    checkpoint_path = save_best_patch_checkpoint(args.diag_save_dir, model, current_iter)
                    print_fn('patch best iter {} updated: checkpoint={}'.format(current_iter, checkpoint_path))

                if trigger_result['triggered']:
                    patch_trigger_iter = current_iter
                    patch_selected_iter = trigger_result['selected_iter']
                    selected_diagnosis_df, selected_score_iter, selected_score_path = resolve_best_patch_diagnosis(
                        diagnosis_df,
                        current_iter,
                        patch_selected_iter,
                        args.diag_save_dir,
                    )
                    post_summary = {
                        'stage': 'patch_selected',
                        'patch_trigger_iter': int(patch_trigger_iter),
                        'patch_selected_iter': None if patch_selected_iter is None else int(patch_selected_iter),
                        'score_source_iter': None if selected_score_iter is None else int(selected_score_iter),
                        'score_source_path': selected_score_path,
                    }
                    post_summary.update(trigger_result['summary'])
                    save_patch_artifacts(iter_dir, diagnosis_df=None, bootstrap_df=None, extra_summary=post_summary)
                    print_fn('patch trigger iter {}: status={}, selected_iter={}'.format(
                        patch_trigger_iter,
                        trigger_result['status'],
                        patch_selected_iter,
                    ))
                    break

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
        print_fn('warmup iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
        if patch_trigger_iter is not None or it == total_iters:
            break

    if args.patch_weight_load_path is not None:
        patch_weight_bank = torch.load(args.patch_weight_load_path, map_location='cpu')
        patch_weight_summary_df = build_patch_weight_summary_from_bank(patch_weight_bank)
        force_all_ones = bool((patch_weight_summary_df['suppress_ratio'] <= 0).all()) if len(patch_weight_summary_df) > 0 else True
        patch_best_active_ratio = 0.0 if force_all_ones else patch_best_active_ratio
    else:
        if best_patch_diagnosis_df is None or best_patch_payload is None:
            raise RuntimeError('patch warmup finished without best patch diagnosis')
        force_all_ones = patch_best_active_ratio is None or float(patch_best_active_ratio) < float(args.patch_min_active_ratio)
        patch_weight_bank, patch_weight_summary_df = build_patch_weight_bank(
            best_patch_diagnosis_df,
            best_patch_payload,
            args=args,
            force_all_ones=force_all_ones,
        )
    patch_weight_summary = {
        'patch_best_iter': None if patch_best_iter is None else int(patch_best_iter),
        'patch_best_U': None if patch_best_U is None else float(patch_best_U),
        'patch_best_SE': None if patch_best_SE is None else float(patch_best_SE),
        'patch_best_active_ratio': None if patch_best_active_ratio is None else float(patch_best_active_ratio),
        'skip_patch_denoise': bool(force_all_ones),
        'patch_weight_mode': args.patch_weight_mode,
        'patch_grid_size': [int(patch_grid_size[0]), int(patch_grid_size[1])],
    }
    bank_path, _ = save_patch_weight_artifacts(
        args.diag_save_dir,
        patch_weight_bank,
        patch_weight_summary_df,
        extra_summary=patch_weight_summary,
        bank_save_path=args.patch_weight_save_path,
    )
    print_fn('patch weight bank saved: {}, skip_patch_denoise={}'.format(bank_path, force_all_ones))

    model, trainable = build_model(device)
    optimizer, lr_scheduler = build_optimizer_and_scheduler(trainable, total_iters=total_iters)
    train_data, train_dataloader = build_patch_train_dataloader(
        train_data_list,
        data_root=args.data_path,
        item_list=item_list,
        patch_weight_bank=patch_weight_bank,
        patch_grid_size=patch_grid_size,
        batch_size=batch_size,
        num_workers=num_workers,
        skip_patch_denoise=force_all_ones,
    )
    print_fn('restart retrain image number:{}'.format(len(train_data)))

    final_eval_summary = None
    it = 0
    for _ in range(int(np.ceil(total_iters / len(train_dataloader)))):
        loss_list = []
        for images, _, meta in train_dataloader:
            loss_value = train_patch_step(
                model,
                trainable,
                optimizer,
                lr_scheduler,
                images,
                meta['patch_weight'],
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
                model.train()

            it += 1
            if it == total_iters:
                break
        print_fn('restart iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
        if it == total_iters:
            break

    total_time = time.time() - train_start_time
    time_per_iter = total_time / max(1, total_iters * 2)
    print_fn('Patch training finished. Total time: {:.2f}s ({:.2f} min)'.format(total_time, total_time / 60.0))
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
    print_fn('  approx_time_per_iter_s   {:.4f}'.format(time_per_iter))


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    parser = argparse.ArgumentParser(description='Standalone patch-level runner for VisA')
    parser.add_argument('--data_path', type=str, default='../VisA_pytorch/1cls')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='vitill_visa_patch')
    parser.add_argument('--diag_save_dir', type=str, default='./patch_diag')
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--diag_batch_size', type=int, default=16)
    parser.add_argument('--diag_num_workers', type=int, default=4)
    parser.add_argument('--diag_max_ratio', type=float, default=0.01)
    parser.add_argument('--diag_resize_mask', type=int, default=256)
    parser.add_argument('--patch_grid_size', type=str, default='32,32')
    parser.add_argument('--patch_use_smoothed_map', action='store_true')
    parser.add_argument('--patch_check_start_ratio', type=float, default=0.01)
    parser.add_argument('--patch_check_end_ratio', type=float, default=0.15)
    parser.add_argument('--patch_check_interval', type=int, default=20)
    parser.add_argument('--patch_tau_bic', type=float, default=10.0)
    parser.add_argument('--patch_tau_sep', type=float, default=1.0)
    parser.add_argument('--patch_tau_conf', type=float, default=0.3)
    parser.add_argument('--patch_pi_min', type=float, default=0.005)
    parser.add_argument('--patch_pi_max', type=float, default=0.30)
    parser.add_argument('--patch_min_mad', type=float, default=1e-6)
    parser.add_argument('--patch_bootstrap_B', type=int, default=20)
    parser.add_argument('--patch_ci_z', type=float, default=1.96)
    parser.add_argument('--patch_improve_eps', type=float, default=1e-6)
    parser.add_argument('--patch_min_active_ratio', type=float, default=0.01)
    parser.add_argument('--patch_min_checks_before_trigger', type=int, default=3)
    parser.add_argument('--patch_min_checks_after_best', type=int, default=1)
    parser.add_argument('--patch_weight_mode', type=str, default='hybrid', choices=['hard', 'soft', 'hybrid'])
    parser.add_argument('--patch_tau_soft', type=float, default=0.5)
    parser.add_argument('--patch_tau_hard', type=float, default=0.95)
    parser.add_argument('--patch_w_min', type=float, default=0.05)
    parser.add_argument('--patch_lambda', type=float, default=0.95)
    parser.add_argument('--patch_max_suppress_ratio', type=float, default=0.30)
    parser.add_argument('--patch_train_mode', type=str, default='restart', choices=['restart'])
    parser.add_argument('--patch_loss_norm', type=str, default='per_image', choices=['per_image'])
    parser.add_argument('--patch_weight_load_path', type=str, default=None)
    parser.add_argument('--patch_weight_save_path', type=str, default=None)
    args = parser.parse_args()

    item_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    if not os.path.isabs(args.diag_save_dir):
        args.diag_save_dir = os.path.join(args.save_dir, args.save_name, args.diag_save_dir)
    if args.patch_weight_save_path is None:
        args.patch_weight_save_path = os.path.join(args.diag_save_dir, 'patch_weight_bank.pt')

    device = 'cuda:{}'.format(args.gpus) if torch.cuda.is_available() else 'cpu'
    print_fn(device)
    train(item_list)
