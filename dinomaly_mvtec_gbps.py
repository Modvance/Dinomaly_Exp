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
    build_train_dataloader,
    build_train_eval_dataloader,
    evaluate_model,
    get_logger,
    rebuild_pruned_train_loaders,
    score_trainset,
    setup_seed,
    should_run_check,
    train_step,
)
from feature_grouping import prepare_gbps_groups
from gbps_runner_utils import (
    build_gbps_hard_delete_plan,
    evaluate_gbps_ci_peak_trigger,
    resolve_gbps_best_scored_df,
    save_gbps_artifacts,
    save_gbps_prune_artifacts,
    summarize_class_label_counts,
    summarize_contamination_labels,
)
from safe_gbps import compute_gbps_u_with_bootstrap
from warmup_diag import load_injected_manifest

warnings = __import__('warnings')
warnings.filterwarnings('ignore')


def train(item_list):
    setup_seed(1)

    total_iters = DEFAULT_TOTAL_ITERS
    batch_size = DEFAULT_BATCH_SIZE
    num_workers = DEFAULT_NUM_WORKERS
    train_start_time = time.time()
    final_eval_summary = None

    train_data_list, test_data_list = build_datasets(args.data_path, item_list)
    contaminated_paths, manifest_path = load_injected_manifest(
        args.data_path,
        os.path.dirname(__file__),
        manifest_path=args.diag_manifest_path,
    )
    if contaminated_paths is None:
        print_fn('gbps diagnosis: contamination manifest not found, contamination-aware logging will be unavailable.')
    else:
        print_fn('gbps diagnosis: loaded contamination manifest {}.'.format(manifest_path))

    train_data, train_dataloader = build_train_dataloader(train_data_list, batch_size=batch_size, num_workers=num_workers)
    train_eval_dataloader = build_train_eval_dataloader(
        train_data_list,
        data_root=args.data_path,
        batch_size=args.diag_batch_size,
        num_workers=args.diag_num_workers,
        item_list=item_list,
        contaminated_paths=contaminated_paths,
    )

    model, trainable = build_model(device)
    optimizer, lr_scheduler = build_optimizer_and_scheduler(trainable, total_iters=total_iters)

    gbps_group_assignments_df, gbps_group_info = prepare_gbps_groups(
        model,
        train_eval_dataloader,
        device,
        args.diag_save_dir,
        args,
    )
    print_fn('gbps groups prepared: selected_k={}, loaded_from_cache={}'.format(
        gbps_group_info.get('selected_k'),
        gbps_group_info.get('loaded_from_cache'),
    ))
    print_fn('train image number:{}'.format(len(train_data)))

    gbps_best_U = None
    gbps_best_SE = None
    gbps_best_iter = None
    gbps_best_noise_evidence = None
    gbps_best_check_count = None
    gbps_check_count = 0
    gbps_trigger_iter = None
    gbps_selected_iter = None
    gbps_has_postprocessed = False

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
            if gbps_trigger_iter is None and should_run_check(
                current_iter,
                total_iters,
                args.gbps_check_start_ratio,
                args.gbps_check_end_ratio,
                args.gbps_check_interval,
            ):
                gbps_check_count += 1
                iter_dir = os.path.join(args.diag_save_dir, 'iter_{:05d}'.format(current_iter))
                scored_df = score_trainset(
                    model,
                    train_eval_dataloader,
                    device,
                    max_ratio=args.diag_max_ratio,
                    resize_mask=args.diag_resize_mask,
                )
                gbps_result = compute_gbps_u_with_bootstrap(
                    scored_df.copy(),
                    gbps_group_assignments_df,
                    B=args.gbps_bootstrap_B,
                    args=args,
                )
                t_max = max(1, int(total_iters * args.gbps_check_end_ratio))
                trigger_result = evaluate_gbps_ci_peak_trigger(
                    U_t=gbps_result['gbps_U'],
                    SE_t=gbps_result['gbps_SE'],
                    noise_evidence=gbps_result['global_noise_evidence'],
                    best_U=gbps_best_U,
                    best_SE=gbps_best_SE,
                    best_iter=gbps_best_iter,
                    best_noise_evidence=gbps_best_noise_evidence,
                    best_check_count=gbps_best_check_count,
                    check_count=gbps_check_count,
                    current_iter=current_iter,
                    ci_z=args.gbps_ci_z,
                    improve_eps=args.gbps_improve_eps,
                    min_checks_before_trigger=args.gbps_min_checks_before_trigger,
                    min_checks_after_best=args.gbps_min_checks_after_best,
                    min_noise_evidence=args.gbps_min_noise_evidence,
                    force_trigger=current_iter >= t_max,
                )
                gbps_best_U = trigger_result['best_U']
                gbps_best_SE = trigger_result['best_SE']
                gbps_best_iter = trigger_result['best_iter']
                gbps_best_noise_evidence = trigger_result['best_noise_evidence']
                gbps_best_check_count = trigger_result['best_check_count']

                summary = dict(gbps_result['summary'])
                summary.update(trigger_result['summary'])
                summary['gbps_group_info'] = gbps_group_info
                summary['diag_manifest_path'] = manifest_path
                summary['diag_manifest_requested_path'] = args.diag_manifest_path
                summary['diag_has_contamination_labels'] = bool(contaminated_paths is not None)
                summary['diag_contamination_label_mode'] = 'manifest' if contaminated_paths is not None else 'unavailable'
                summary.update(summarize_contamination_labels(scored_df))
                summary['class_label_counts'] = summarize_class_label_counts(scored_df)
                save_gbps_artifacts(
                    iter_dir,
                    group_metrics_df=gbps_result['group_metrics_df'],
                    bootstrap_df=gbps_result['bootstrap_df'],
                    sample_group_scores_df=gbps_result['sample_group_scores_df'],
                    extra_summary=summary,
                    scored_df=scored_df,
                )

                if trigger_result['triggered']:
                    gbps_trigger_iter = current_iter
                    gbps_selected_iter = trigger_result['selected_iter']
                    selected_scored_df, selected_score_iter, selected_score_path = resolve_gbps_best_scored_df(
                        scored_df,
                        current_iter,
                        gbps_selected_iter,
                        args.diag_save_dir,
                    )
                    postprocess_summary = {
                        'stage': 'gbps_selected',
                        'gbps_trigger_iter': int(gbps_trigger_iter),
                        'gbps_selected_iter': None if gbps_selected_iter is None else int(gbps_selected_iter),
                        'score_source_iter': None if selected_score_iter is None else int(selected_score_iter),
                        'score_source_path': selected_score_path,
                        'selected_iter_less_than_trigger_iter': bool(gbps_selected_iter is not None and gbps_selected_iter < gbps_trigger_iter),
                        'gbps_postprocess_mode': args.gbps_postprocess_mode,
                        'diag_manifest_path': manifest_path,
                        'diag_manifest_requested_path': args.diag_manifest_path,
                        'diag_has_contamination_labels': bool(contaminated_paths is not None),
                        'diag_contamination_label_mode': 'manifest' if contaminated_paths is not None else 'unavailable',
                    }
                    if args.gbps_postprocess_mode == 'remove' and not gbps_has_postprocessed and selected_scored_df is not None:
                        train_image_count_before_prune = len(train_data)
                        delete_plan = build_gbps_hard_delete_plan(
                            selected_scored_df,
                            gbps_group_assignments_df,
                            prune_ratio=args.gbps_prune_ratio,
                            min_keep_per_group=args.gbps_min_keep_per_group,
                        )
                        rebuild_result = rebuild_pruned_train_loaders(
                            train_data_list,
                            delete_plan['retained_index_map'],
                            data_root=args.data_path,
                            item_list=item_list,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            diag_batch_size=args.diag_batch_size,
                            diag_num_workers=args.diag_num_workers,
                            contaminated_paths=contaminated_paths,
                        )
                        train_data_list = rebuild_result['train_data_list']
                        train_data = rebuild_result['train_data']
                        train_dataloader = rebuild_result['train_dataloader']
                        train_eval_dataloader = rebuild_result['train_eval_dataloader']
                        gbps_has_postprocessed = True
                        postprocess_summary.update({
                            'gbps_pruned_count': int(delete_plan['summary']['num_pruned']),
                            'gbps_retained_count': int(delete_plan['summary']['num_retained']),
                            'gbps_prune_ratio': float(args.gbps_prune_ratio),
                            'gbps_min_keep_per_group': int(args.gbps_min_keep_per_group),
                            'gbps_prune_group_counts': delete_plan['summary']['group_counts'],
                            'train_image_count_before_prune': int(train_image_count_before_prune),
                            'train_image_count_after_prune': int(len(train_data)),
                        })
                        save_gbps_prune_artifacts(iter_dir, delete_plan, extra_summary=postprocess_summary)
                        print_fn('gbps remove iter {}: pruned {} / {} training samples, train image number {} -> {}'.format(
                            gbps_selected_iter,
                            delete_plan['summary']['num_pruned'],
                            delete_plan['summary']['num_samples_before_prune'],
                            train_image_count_before_prune,
                            len(train_data),
                        ))
                    save_gbps_artifacts(
                        iter_dir,
                        group_metrics_df=None,
                        bootstrap_df=None,
                        extra_summary=postprocess_summary,
                    )
                    print_fn('gbps trigger iter {}: status={}, selected_iter={}'.format(
                        gbps_trigger_iter,
                        trigger_result['status'],
                        gbps_selected_iter,
                    ))

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
        print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))

    total_time = time.time() - train_start_time
    time_per_iter = total_time / total_iters
    iters_per_sec = total_iters / total_time
    samples_per_sec = batch_size / time_per_iter

    print_fn('Training finished. Total time: {:.2f}s ({:.2f} min), {:.4f} s/iter'.format(total_time, total_time / 60.0, time_per_iter))
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

    parser = argparse.ArgumentParser(description='Standalone Safe-GBPS runner for MVTec')
    parser.add_argument('--data_path', type=str, default='../mvtec_anomaly_detection')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='vitill_mvtec_gbps')
    parser.add_argument('--diag_save_dir', type=str, default='./gbps_diag')
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--eval_interval', type=int, default=5000)
    parser.add_argument('--diag_batch_size', type=int, default=16)
    parser.add_argument('--diag_num_workers', type=int, default=4)
    parser.add_argument('--diag_max_ratio', type=float, default=0.01)
    parser.add_argument('--diag_resize_mask', type=int, default=256)
    parser.add_argument('--diag_manifest_path', type=str, default=None)
    parser.add_argument('--gbps_check_start_ratio', type=float, default=0.01)
    parser.add_argument('--gbps_check_end_ratio', type=float, default=0.15)
    parser.add_argument('--gbps_check_interval', type=int, default=20)
    parser.add_argument('--gbps_grouping_method', type=str, default='kmeans')
    parser.add_argument('--gbps_num_groups', type=int, default=None)
    parser.add_argument('--gbps_auto_k', action='store_true')
    parser.add_argument('--gbps_k_candidates', type=str, default='8,10,12,15,20')
    parser.add_argument('--gbps_pca_dim', type=int, default=64)
    parser.add_argument('--gbps_min_group_size', type=int, default=8)
    parser.add_argument('--gbps_embedding_source', type=str, default='encoder')
    parser.add_argument('--gbps_group_cache_path', type=str, default=None)
    parser.add_argument('--gbps_tau_bic', type=float, default=0.0)
    parser.add_argument('--gbps_tau_sep', type=float, default=0.5)
    parser.add_argument('--gbps_tau_conf', type=float, default=0.1)
    parser.add_argument('--gbps_pi_min', type=float, default=0.05)
    parser.add_argument('--gbps_pi_max', type=float, default=0.95)
    parser.add_argument('--gbps_gate_mode', type=str, default='hard')
    parser.add_argument('--gbps_min_noise_evidence', type=float, default=0.05)
    parser.add_argument('--gbps_postprocess_mode', type=str, default='none', choices=['none', 'remove'])
    parser.add_argument('--gbps_prune_ratio', type=float, default=0.1)
    parser.add_argument('--gbps_min_keep_per_group', type=int, default=8)
    parser.add_argument('--gbps_bootstrap_B', type=int, default=20)
    parser.add_argument('--gbps_ci_z', type=float, default=1.96)
    parser.add_argument('--gbps_improve_eps', type=float, default=1e-6)
    parser.add_argument('--gbps_min_checks_before_trigger', type=int, default=2)
    parser.add_argument('--gbps_min_checks_after_best', type=int, default=1)
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
