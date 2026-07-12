import argparse
import json
import os
import time

import numpy as np
import torch

from dinomaly_train_base import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CROP_SIZE,
    DEFAULT_ENCODER_NAME,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_TOTAL_ITERS,
    build_datasets,
    build_model,
    build_optimizer_and_scheduler,
    build_train_dataloader,
    build_train_eval_dataloader,
    evaluate_model,
    get_logger,
    load_model_from_train_checkpoint,
    rebuild_pruned_train_loaders,
    save_train_checkpoint,
    score_trainset,
    setup_seed,
    should_run_check,
    train_step,
)
from gbps_runner_utils import (
    evaluate_gbps_ci_peak_trigger,
    resolve_gbps_best_scored_df,
    save_gbps_artifacts,
    summarize_class_label_counts,
    summarize_contamination_labels,
)
from tailguard_artifacts import (
    save_tailguard_iteration_artifacts,
    save_tailguard_memory_artifacts,
    save_tailguard_prune_artifacts,
    save_tailguard_summary,
)
from tailguard_gbps import (
    aggregate_tailguard_risk_history,
    build_tailguard_delete_plan,
    run_tailguard_gbps_iteration,
)
from tailguard_memory import build_tail_group_memory_bank, run_tail_group_memory_evaluation
from tailguard_prepare import prepare_tailguard_training_metadata
from warmup_diag import load_injected_manifest

warnings = __import__('warnings')
warnings.filterwarnings('ignore')


MVTec_ITEM_LIST = [
    'carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
    'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper',
]


def _build_default_tail_support_df(prepare_result):
    metadata_df = prepare_result['train_metadata_df'].copy()
    support_df = metadata_df.loc[metadata_df['tail_candidate'] == 1].copy().reset_index(drop=True)
    support_df['aggregated_final_gbps_risk'] = 0.0
    return support_df


def _evaluate_current_model(model, test_data_list, item_list, device, args, print_fn=None):
    return evaluate_model(
        model,
        test_data_list,
        item_list,
        device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        max_ratio=float(args.max_ratio),
        resize_mask=int(args.resize_mask),
        print_fn=print_fn,
    )


def train(item_list):
    setup_seed(1)

    total_iters = int(args.total_iters)
    batch_size = int(args.batch_size)
    num_workers = int(args.num_workers)
    train_start_time = time.time()
    final_eval_summary = None
    memory_eval_summary = None
    metrics_by_mode = None
    memory_saved = None
    delete_plan = None
    gbps_trigger_summary = None

    if args.checkpoint_path:
        model, trainable, checkpoint_metadata = load_model_from_train_checkpoint(args.checkpoint_path, device)
        image_size = int(checkpoint_metadata['image_size'])
        crop_size = int(checkpoint_metadata['crop_size'])
        print_fn('loaded checkpoint: {}'.format(args.checkpoint_path))
    else:
        model, trainable = build_model(device, encoder_name=args.encoder_name)
        checkpoint_metadata = None
        image_size = int(args.image_size)
        crop_size = int(args.crop_size)
        print_fn('built model from encoder: {}'.format(args.encoder_name))

    optimizer, lr_scheduler = build_optimizer_and_scheduler(
        trainable,
        total_iters=total_iters,
        lr=float(args.lr),
        final_lr=float(args.final_lr),
        warmup_iters=int(args.warmup_iters),
        weight_decay=float(args.weight_decay),
    )

    train_data_list, test_data_list = build_datasets(
        args.data_path,
        item_list,
        image_size=image_size,
        crop_size=crop_size,
    )
    contaminated_paths, manifest_path = load_injected_manifest(
        args.data_path,
        os.path.dirname(__file__),
        manifest_path=args.diag_manifest_path,
    )
    if contaminated_paths is None:
        print_fn('tailguard diagnosis: contamination manifest not found, contamination-aware logging will be unavailable.')
    else:
        print_fn('tailguard diagnosis: loaded contamination manifest {}.'.format(manifest_path))

    train_data, train_dataloader = build_train_dataloader(
        train_data_list,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    train_eval_dataloader = build_train_eval_dataloader(
        train_data_list,
        data_root=args.data_path,
        batch_size=args.diag_batch_size,
        num_workers=args.diag_num_workers,
        item_list=item_list,
        contaminated_paths=contaminated_paths,
    )

    prepare_output_dir = os.path.join(args.save_dir, args.save_name, 'tailguard_prepare')
    prepare_result = prepare_tailguard_training_metadata(
        model,
        train_eval_dataloader,
        device,
        prepare_output_dir,
        args,
    )
    print_fn('tailguard prepare: tail_candidates={}, groups={}'.format(
        prepare_result['summary']['num_tail_candidates'],
        prepare_result['summary']['num_groups'],
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
    last_eval_iter = None

    it = 0
    for _ in range(int(np.ceil(total_iters / len(train_dataloader)))):
        loss_list = []
        for batch in train_dataloader:
            images, _ = batch if len(batch) == 2 else batch[:2]
            loss_value = train_step(
                model,
                trainable,
                optimizer,
                lr_scheduler,
                images,
                current_zero_based_iter=it,
                sample_weight=None,
                grad_clip_norm=float(args.grad_clip_norm),
            )
            loss_list.append(loss_value)

            current_iter = it + 1
            if gbps_trigger_iter is None and should_run_check(
                current_iter,
                total_iters,
                float(args.gbps_check_start_ratio),
                float(args.gbps_check_end_ratio),
                int(args.gbps_check_interval),
            ):
                gbps_check_count += 1
                iter_dir = os.path.join(args.diag_save_dir, 'iter_{:05d}'.format(current_iter))
                scored_df = score_trainset(
                    model,
                    train_eval_dataloader,
                    device,
                    max_ratio=float(args.diag_max_ratio),
                    resize_mask=int(args.diag_resize_mask),
                )
                tailguard_result = run_tailguard_gbps_iteration(
                    scored_df.copy(),
                    prepare_result['group_assignments_df'],
                    prepare_result['train_metadata_df'],
                    args,
                )
                t_max = max(1, int(total_iters * float(args.gbps_check_end_ratio)))
                trigger_result = evaluate_gbps_ci_peak_trigger(
                    U_t=tailguard_result['gbps_U'],
                    SE_t=tailguard_result['gbps_SE'],
                    noise_evidence=tailguard_result['global_noise_evidence'],
                    best_U=gbps_best_U,
                    best_SE=gbps_best_SE,
                    best_iter=gbps_best_iter,
                    best_noise_evidence=gbps_best_noise_evidence,
                    best_check_count=gbps_best_check_count,
                    check_count=gbps_check_count,
                    current_iter=current_iter,
                    ci_z=float(args.gbps_ci_z),
                    improve_eps=float(args.gbps_improve_eps),
                    min_checks_before_trigger=int(args.gbps_min_checks_before_trigger),
                    min_checks_after_best=int(args.gbps_min_checks_after_best),
                    min_noise_evidence=float(args.gbps_min_noise_evidence),
                    force_trigger=current_iter >= t_max,
                )
                gbps_best_U = trigger_result['best_U']
                gbps_best_SE = trigger_result['best_SE']
                gbps_best_iter = trigger_result['best_iter']
                gbps_best_noise_evidence = trigger_result['best_noise_evidence']
                gbps_best_check_count = trigger_result['best_check_count']

                summary = dict(tailguard_result['summary'])
                summary.update(trigger_result['summary'])
                summary['tailguard_prepare_summary'] = prepare_result['summary']
                summary['diag_manifest_path'] = manifest_path
                summary['diag_manifest_requested_path'] = args.diag_manifest_path
                summary['diag_has_contamination_labels'] = bool(contaminated_paths is not None)
                summary['diag_contamination_label_mode'] = 'manifest' if contaminated_paths is not None else 'unavailable'
                summary.update(summarize_contamination_labels(scored_df))
                summary['class_label_counts'] = summarize_class_label_counts(scored_df)
                save_gbps_artifacts(
                    iter_dir,
                    group_metrics_df=tailguard_result['group_metrics_df'],
                    bootstrap_df=tailguard_result['bootstrap_df'],
                    sample_group_scores_df=tailguard_result['sample_group_scores_df'],
                    extra_summary=summary,
                    scored_df=scored_df,
                )
                save_tailguard_iteration_artifacts(
                    iter_dir,
                    tailguard_result['global_group_metrics_df'],
                    tailguard_result['sample_tailguard_scores_df'],
                    extra_summary={
                        'tailguard_local_risk_mean': tailguard_result['summary']['tailguard_local_risk_mean'],
                        'tailguard_global_group_risk_mean': tailguard_result['summary']['tailguard_global_group_risk_mean'],
                        'tailguard_final_risk_mean': tailguard_result['summary']['tailguard_final_risk_mean'],
                        'tailguard_num_global_active_groups': tailguard_result['summary']['tailguard_num_global_active_groups'],
                    },
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
                        'stage': 'tailguard_selected',
                        'gbps_trigger_iter': int(gbps_trigger_iter),
                        'gbps_selected_iter': None if gbps_selected_iter is None else int(gbps_selected_iter),
                        'score_source_iter': None if selected_score_iter is None else int(selected_score_iter),
                        'score_source_path': selected_score_path,
                        'selected_iter_less_than_trigger_iter': bool(gbps_selected_iter is not None and gbps_selected_iter < gbps_trigger_iter),
                        'tailguard_postprocess_mode': args.gbps_postprocess_mode,
                        'diag_manifest_path': manifest_path,
                        'diag_manifest_requested_path': args.diag_manifest_path,
                        'diag_has_contamination_labels': bool(contaminated_paths is not None),
                        'diag_contamination_label_mode': 'manifest' if contaminated_paths is not None else 'unavailable',
                    }
                    if args.gbps_postprocess_mode == 'remove' and not gbps_has_postprocessed and selected_scored_df is not None:
                        aggregated_risk_df = aggregate_tailguard_risk_history(
                            args.diag_save_dir,
                            gbps_selected_iter,
                            lookback_checks=int(args.tg_risk_lookback_checks),
                        )
                        train_image_count_before_prune = len(train_data)
                        delete_plan = build_tailguard_delete_plan(
                            selected_scored_df,
                            prepare_result['group_assignments_df'],
                            prepare_result['train_metadata_df'],
                            aggregated_risk_df,
                            args,
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
                            sampler=None,
                            epoch_num_samples=None,
                            phase6_plan=None,
                            use_phase6_meta=False,
                        )
                        train_data_list = rebuild_result['train_data_list']
                        train_data = rebuild_result['train_data']
                        train_dataloader = rebuild_result['train_dataloader']
                        train_eval_dataloader = rebuild_result['train_eval_dataloader']
                        gbps_has_postprocessed = True
                        postprocess_summary.update({
                            'tailguard_pruned_count': int(delete_plan['summary']['num_pruned']),
                            'tailguard_retained_count': int(delete_plan['summary']['num_retained']),
                            'tailguard_tail_support_count': int(delete_plan['summary']['num_tail_support']),
                            'tailguard_tail_suspicious_count': int(delete_plan['summary']['num_tail_suspicious']),
                            'train_image_count_before_prune': int(train_image_count_before_prune),
                            'train_image_count_after_prune': int(len(train_data)),
                        })
                        save_tailguard_prune_artifacts(iter_dir, delete_plan, extra_summary=postprocess_summary)
                        print_fn('tailguard remove iter {}: pruned {} / {} training samples, train image number {} -> {}'.format(
                            gbps_selected_iter,
                            delete_plan['summary']['num_pruned'],
                            delete_plan['summary']['num_samples_before_prune'],
                            train_image_count_before_prune,
                            len(train_data),
                        ))
                    save_tailguard_iteration_artifacts(
                        iter_dir,
                        tailguard_result['global_group_metrics_df'],
                        tailguard_result['sample_tailguard_scores_df'],
                        extra_summary=postprocess_summary,
                    )
                    gbps_trigger_summary = dict(postprocess_summary)
                    print_fn('tailguard trigger iter {}: status={}, selected_iter={}'.format(
                        gbps_trigger_iter,
                        trigger_result['status'],
                        gbps_selected_iter,
                    ))

            if current_iter % int(args.eval_interval) == 0:
                final_eval_summary = _evaluate_current_model(model, test_data_list, item_list, device, args, print_fn=print_fn)
                last_eval_iter = int(current_iter)
                model.train()

            it += 1
            if it == total_iters:
                break
        print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))

    if last_eval_iter != int(it):
        final_eval_summary = _evaluate_current_model(model, test_data_list, item_list, device, args, print_fn=print_fn)
        last_eval_iter = int(it)
        model.train()

    memory_bank = {}
    if bool(args.tg_memory_enable):
        support_df = _build_default_tail_support_df(prepare_result) if delete_plan is None else delete_plan['tail_support_samples'].copy()
        memory_bank = build_tail_group_memory_bank(
            model,
            train_eval_dataloader,
            device,
            support_df,
            args,
        )
        print_fn('tail-group memory: built {} groups from {} support samples'.format(len(memory_bank), len(support_df)))
        score_df, per_class_metrics, memory_eval_summary, metrics_by_mode = run_tail_group_memory_evaluation(
            model,
            test_data_list,
            item_list,
            memory_bank,
            args,
            device,
        )
        memory_output_dir = os.path.join(args.save_dir, args.save_name, args.tg_memory_artifact_dir)
        memory_metadata = {
            'checkpoint_path': args.checkpoint_path,
            'encoder_name': checkpoint_metadata['encoder_name'] if checkpoint_metadata is not None else args.encoder_name,
            'image_size': image_size,
            'crop_size': crop_size,
            'tailguard_memory_groups': int(len(memory_bank)),
            'tailguard_memory_support_samples': int(len(support_df)),
            'tg_mem_min_support_images': int(args.tg_mem_min_support_images),
            'tg_mem_gate_quantile': float(args.tg_mem_gate_quantile),
            'tg_mem_gate_mad_scale': float(args.tg_mem_gate_mad_scale),
            'tg_mem_fusion_lambda': float(args.tg_mem_fusion_lambda),
            'tg_mem_topk_ratio': float(args.tg_mem_topk_ratio),
        }
        memory_saved = save_tailguard_memory_artifacts(
            memory_output_dir,
            memory_bank,
            score_df,
            per_class_metrics,
            memory_eval_summary,
            memory_metadata,
            metrics_by_mode=metrics_by_mode,
        )
        print_fn('saved tail-group memory artifacts to {}'.format(memory_output_dir))
        print_fn(json.dumps(memory_eval_summary, indent=2, ensure_ascii=False))

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

    checkpoint_path = save_train_checkpoint(
        args.save_dir,
        args.save_name,
        model,
        iteration=it,
        args=args,
        final_eval_summary=final_eval_summary,
    )
    print_fn('saved final model checkpoint to {}'.format(checkpoint_path))

    tailguard_summary = {
        'checkpoint_path': checkpoint_path,
        'input_checkpoint_path': args.checkpoint_path,
        'prepare_summary': prepare_result['summary'],
        'gbps_trigger_summary': gbps_trigger_summary,
        'final_eval_summary': final_eval_summary,
        'memory_eval_summary': memory_eval_summary,
        'memory_artifacts': memory_saved,
        'total_time_s': float(total_time),
        'time_per_iter_s': float(time_per_iter),
        'iters_per_sec': float(iters_per_sec),
        'samples_per_sec': float(samples_per_sec),
        'tailguard_memory_groups': int(len(memory_bank)),
    }
    summary_path = save_tailguard_summary(os.path.join(args.save_dir, args.save_name), tailguard_summary)
    print_fn('saved tailguard summary to {}'.format(summary_path))


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    parser = argparse.ArgumentParser(description='Standalone TailGuard runner for Dinomaly MVTec')
    parser.add_argument('--data_path', type=str, default='../mvtec_anomaly_detection')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='vitill_mvtec_tailguard')
    parser.add_argument('--diag_save_dir', type=str, default='tailguard_diag')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--total_iters', type=int, default=DEFAULT_TOTAL_ITERS)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument('--eval_interval', type=int, default=5000)
    parser.add_argument('--image_size', type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument('--crop_size', type=int, default=DEFAULT_CROP_SIZE)
    parser.add_argument('--encoder_name', type=str, default=DEFAULT_ENCODER_NAME)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--final_lr', type=float, default=2e-4)
    parser.add_argument('--warmup_iters', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--max_ratio', type=float, default=0.01)
    parser.add_argument('--resize_mask', type=int, default=256)
    parser.add_argument('--diag_batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--diag_num_workers', type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument('--diag_max_ratio', type=float, default=0.01)
    parser.add_argument('--diag_resize_mask', type=int, default=256)
    parser.add_argument('--diag_manifest_path', type=str, default=None)

    parser.add_argument('--tailsampler_type', type=str, default='adaptive', choices=['tail', 'adaptive', 'adaptive_trim_mode'])
    parser.add_argument('--tailsampler_th_type', type=str, default=None)
    parser.add_argument('--tailsampler_vote_type', type=str, default=None)
    parser.add_argument('--tailsampler_percentile', type=float, default=0.15)
    parser.add_argument('--tailsampler_embedding_source', type=str, default='encoder', choices=['encoder', 'encoder_cls'])
    parser.add_argument('--tg_tail_embedding_source', type=str, default='encoder_cls', choices=['encoder', 'encoder_cls'])
    parser.add_argument('--tg_save_cls_embeddings', action='store_true')

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
    parser.add_argument('--gbps_tau_bic', type=float, default=0.0)
    parser.add_argument('--gbps_tau_sep', type=float, default=0.5)
    parser.add_argument('--gbps_tau_conf', type=float, default=0.1)
    parser.add_argument('--gbps_pi_min', type=float, default=0.05)
    parser.add_argument('--gbps_pi_max', type=float, default=0.95)
    parser.add_argument('--gbps_gate_mode', type=str, default='hard')
    parser.add_argument('--gbps_min_noise_evidence', type=float, default=0.05)
    parser.add_argument('--gbps_postprocess_mode', type=str, default='remove', choices=['none', 'remove'])
    parser.add_argument('--gbps_prune_ratio', type=float, default=0.1)
    parser.add_argument('--gbps_min_keep_per_group', type=int, default=8)
    parser.add_argument('--gbps_bootstrap_B', type=int, default=20)
    parser.add_argument('--gbps_ci_z', type=float, default=1.96)
    parser.add_argument('--gbps_improve_eps', type=float, default=1e-6)
    parser.add_argument('--gbps_min_checks_before_trigger', type=int, default=2)
    parser.add_argument('--gbps_min_checks_after_best', type=int, default=1)

    parser.add_argument('--tg_enable_global_group_risk', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--tg_global_tau_bic', type=float, default=0.0)
    parser.add_argument('--tg_global_tau_sep', type=float, default=0.5)
    parser.add_argument('--tg_global_tau_conf', type=float, default=0.1)
    parser.add_argument('--tg_global_pi_min', type=float, default=0.05)
    parser.add_argument('--tg_global_pi_max', type=float, default=0.95)
    parser.add_argument('--tg_risk_lookback_checks', type=int, default=3)
    parser.add_argument('--tg_tail_risk_remove_thr', type=float, default=0.5)

    parser.add_argument('--tg_memory_enable', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--tg_memory_artifact_dir', type=str, default='tail_group_memory')
    parser.add_argument('--tg_mem_min_support_images', type=int, default=1)
    parser.add_argument('--tg_mem_gate_quantile', type=float, default=0.9)
    parser.add_argument('--tg_mem_gate_mad_scale', type=float, default=1.0)
    parser.add_argument('--tg_mem_fusion_lambda', type=float, default=1.0)
    parser.add_argument('--tg_mem_topk_ratio', type=float, default=0.05)
    parser.add_argument('--tg_mem_chunk_size', type=int, default=4096)
    parser.add_argument('--mem_max_patches_per_class', type=int, default=20000)
    args = parser.parse_args()

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    if not os.path.isabs(args.diag_save_dir):
        args.diag_save_dir = os.path.join(args.save_dir, args.save_name, args.diag_save_dir)

    device = 'cuda:{}'.format(args.gpus) if torch.cuda.is_available() else 'cpu'
    print_fn(device)
    train(MVTec_ITEM_LIST)
