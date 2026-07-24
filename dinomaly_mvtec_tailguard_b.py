import argparse
import json
import os
import time

import numpy as np
import pandas as pd
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
    _checkpoint_args_get,
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
from gbps_runner_utils import evaluate_gbps_ci_peak_trigger, resolve_gbps_best_scored_df
from tailguard_b_artifacts import (
    save_tailguard_b_attachment_replay_config,
    save_tailguard_b_attachment_artifacts,
    save_tailguard_b_denoising_diagnostics,
    save_tailguard_b_pseudoclass_artifacts,
    save_tailguard_b_pseudoclass_report_artifacts,
    save_tailguard_b_gbps_iteration_artifacts,
    save_tailguard_b_memory_artifacts,
    save_tailguard_b_stage2_artifacts,
    save_tailguard_b_summary,
    save_tailguard_b_trigger_summary,
)
from tailguard_b_attachment import build_tail_attachment_plan
from tailguard_b_gbps import (
    build_tailguard_b_head_delete_plan,
    build_tailguard_b_stage2_plan,
    classify_tail_attached_stable_risk,
    load_tailguard_b_selected_checkpoint,
    run_tailguard_b_gbps_iteration,
    save_tailguard_b_selected_checkpoint,
)
from tailguard_b_diagnostics import (
    build_tailguard_b_denoising_diagnostics,
    build_tailguard_b_pseudoclass_report,
)
from tailguard_b_memory import (
    build_tailguard_b_pseudoclass_memory_system,
    run_pseudoclass_memory_evaluation,
)
from tailguard_b_dataset_profiles import (
    MVTec_PROFILE,
    dataset_profile_names,
    get_dataset_profile,
    profile_provenance,
    validate_profile_contract,
)
from tailguard_b_prepare import prepare_tailguard_b_metadata
from tailguard_b_pseudoclasses import build_tailguard_b_pseudoclass_registry
from warmup_diag import load_injected_manifest

warnings = __import__('warnings')
warnings.filterwarnings('ignore')


MVTec_ITEM_LIST = list(MVTec_PROFILE.item_list)


def _build_retained_index_map(samples_df: pd.DataFrame, num_classes=None):
    retained_index_map = {}
    if num_classes is not None:
        for class_id in range(int(num_classes)):
            retained_index_map[int(class_id)] = []
    for class_id, class_df in samples_df.groupby('class_id', sort=True):
        retained_index_map[int(class_id)] = [int(value) for value in class_df['base_idx'].astype(int).tolist()]
    return retained_index_map


def _validate_checkpoint_profile(profile, checkpoint_metadata):
    checkpoint_args = checkpoint_metadata.get('args') or {}
    validate_profile_contract(
        profile,
        _checkpoint_args_get(checkpoint_args, 'dataset_profile', None),
        _checkpoint_args_get(checkpoint_args, 'dataset_item_list', None),
        'checkpoint',
    )


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


def _print_eval_summary(summary, print_fn):
    if summary is None:
        return
    print_fn('I-AUROC      {:.2f}'.format(summary['I-AUROC']))
    print_fn('I-AP         {:.2f}'.format(summary['I-AP']))
    print_fn('I-F1         {:.2f}'.format(summary['I-F1']))
    print_fn('P-AUROC      {:.2f}'.format(summary['P-AUROC']))
    print_fn('P-AP         {:.2f}'.format(summary['P-AP']))
    print_fn('P-F1         {:.2f}'.format(summary['P-F1']))
    print_fn('P-AUPRO      {:.2f}'.format(summary['P-AUPRO']))


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
    stage2_plan = None
    attachment_saved = None
    stage2_saved = None
    denoising_diagnostics_artifacts = None
    denoising_diagnostics_summary = None
    pseudoclass_registry = None
    pseudoclass_artifacts = None
    pseudoclass_report_artifacts = None
    pseudoclass_status = 'not_performed'
    pseudoclass_reason = None
    gbps_trigger_summary = None
    selected_checkpoint_path = os.path.join(args.tgb_checkpoint_dir, 'tailguard_b_selected_checkpoint.pt')

    if args.checkpoint_path:
        model, trainable, checkpoint_metadata = load_model_from_train_checkpoint(args.checkpoint_path, device)
        _validate_checkpoint_profile(get_dataset_profile(args.dataset_profile), checkpoint_metadata)
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

    base_train_data_list, test_data_list = build_datasets(
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
        print_fn('tailguard-b diagnosis: contamination manifest not found, contamination-aware logging will be unavailable.')
    else:
        print_fn('tailguard-b diagnosis: loaded contamination manifest {}.'.format(manifest_path))

    full_train_data, full_train_dataloader = build_train_dataloader(
        base_train_data_list,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    full_train_eval_dataloader = build_train_eval_dataloader(
        base_train_data_list,
        data_root=args.data_path,
        batch_size=args.diag_batch_size,
        num_workers=args.diag_num_workers,
        item_list=item_list,
        contaminated_paths=contaminated_paths,
    )

    prepare_result = prepare_tailguard_b_metadata(
        model,
        full_train_eval_dataloader,
        device,
        args.tgb_prepare_dir,
        args,
        manifest_path=manifest_path,
        manifest_requested_path=args.diag_manifest_path,
    )
    print_fn('tailguard-b prepare: T={}, H={}, head_groups={}'.format(
        prepare_result['summary']['num_tail_candidates'],
        prepare_result['summary']['num_head_candidates'],
        prepare_result['summary']['num_head_groups'],
    ))
    analysis_summary = prepare_result['summary'].get('tail_sampler_analysis_only_summary')
    if analysis_summary is not None and 'selection_precision' in analysis_summary:
        print_fn('tail sampler analysis-only: precision={:.4f}, recall={:.4f}, f1={:.4f}, noisy_proxy_rate={:.4f}'.format(
            float(analysis_summary['selection_precision']),
            float(analysis_summary['selection_recall']),
            float(analysis_summary['selection_f1']),
            float(analysis_summary['selected_noise_proxy_rate']),
        ))
    elif analysis_summary is not None:
        print_fn('tail sampler analysis-only skipped: {}'.format(analysis_summary.get('error', analysis_summary.get('status', 'unknown'))))

    full_metadata_df = prepare_result['full_metadata_df'].copy()
    method_metadata_df = full_metadata_df.drop(columns=['is_contaminated', 'class_name'], errors='ignore')
    stage1_training_scope = args.tgb_stage1_training_scope
    if stage1_training_scope == 'h_only':
        h_metadata_df = method_metadata_df.loc[
            method_metadata_df['tail_candidate'].astype(int) == 0
        ].copy().reset_index(drop=True)
        h_index_map = _build_retained_index_map(h_metadata_df, num_classes=len(base_train_data_list))
        stage1_loaders = rebuild_pruned_train_loaders(
            base_train_data_list,
            h_index_map,
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
        train_data_list = stage1_loaders['train_data_list']
        train_data = stage1_loaders['train_data']
        train_dataloader = stage1_loaders['train_dataloader']
        train_eval_dataloader = stage1_loaders['train_eval_dataloader']
    else:
        train_data_list = base_train_data_list
        train_data = full_train_data
        train_dataloader = full_train_dataloader
        train_eval_dataloader = full_train_eval_dataloader
    memory_train_eval_dataloader = full_train_eval_dataloader
    print_fn('stage1 training scope={} train image number:{} / full {}'.format(
        stage1_training_scope,
        len(train_data),
        len(full_train_data),
    ))

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
    while it < total_iters:
        loss_list = []
        reset_loader = False
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

            if gbps_trigger_iter is None and not gbps_has_postprocessed and should_run_check(
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
                    full_train_eval_dataloader,
                    device,
                    max_ratio=float(args.diag_max_ratio),
                    resize_mask=int(args.diag_resize_mask),
                )
                gbps_result = run_tailguard_b_gbps_iteration(
                    scored_df,
                    prepare_result['head_group_assignments_df'],
                    method_metadata_df,
                    args,
                )
                t_max = max(1, int(total_iters * float(args.gbps_check_end_ratio)))
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

                summary = dict(gbps_result['summary'])
                summary.update(trigger_result['summary'])
                summary['prepare_summary'] = prepare_result['summary']
                summary['diag_manifest_path'] = manifest_path
                summary['diag_manifest_requested_path'] = args.diag_manifest_path
                summary['diag_has_contamination_labels'] = bool(contaminated_paths is not None)
                save_tailguard_b_gbps_iteration_artifacts(
                    iter_dir,
                    gbps_result['train_scores_df'],
                    gbps_result['h_group_metrics_df'],
                    gbps_result['h_sample_group_scores_df'],
                    gbps_result['bootstrap_df'],
                    summary,
                )

                if bool(trigger_result['summary']['gbps_improved']):
                    save_tailguard_b_selected_checkpoint(
                        selected_checkpoint_path,
                        model,
                        optimizer,
                        lr_scheduler,
                        iteration=current_iter,
                        gbps_U=gbps_result['gbps_U'],
                        gbps_SE=gbps_result['gbps_SE'],
                        args=args,
                    )
                    print_fn('tailguard-b selected checkpoint updated at iter {}: U={:.6f}, SE={:.6f}'.format(
                        current_iter,
                        float(gbps_result['gbps_U']),
                        float(gbps_result['gbps_SE']),
                    ))

                if trigger_result['triggered']:
                    gbps_trigger_iter = current_iter
                    gbps_selected_iter = trigger_result['selected_iter']
                    selected_scored_df, selected_score_iter, selected_score_path = resolve_gbps_best_scored_df(
                        gbps_result['train_scores_df'],
                        current_iter,
                        gbps_selected_iter,
                        args.diag_save_dir,
                    )
                    gbps_trigger_summary = {
                        'stage': 'tailguard_b_selected',
                        'gbps_trigger_iter': int(gbps_trigger_iter),
                        'gbps_selected_iter': None if gbps_selected_iter is None else int(gbps_selected_iter),
                        'score_source_iter': None if selected_score_iter is None else int(selected_score_iter),
                        'score_source_path': selected_score_path,
                        'selected_checkpoint_path': selected_checkpoint_path,
                        'gbps_status': trigger_result['status'],
                    }
                    save_tailguard_b_trigger_summary(args.diag_save_dir, gbps_trigger_summary)

                    if args.gbps_postprocess_mode == 'remove' and selected_scored_df is not None and os.path.isfile(selected_checkpoint_path):
                        head_delete_plan = build_tailguard_b_head_delete_plan(
                            selected_scored_df,
                            prepare_result['head_group_assignments_df'],
                            method_metadata_df,
                            args,
                        )
                        selected_scores_df = head_delete_plan['selected_scores_df']
                        tail_df = selected_scores_df.loc[selected_scores_df['tail_candidate'].astype(int) == 1].copy().reset_index(drop=True)
                        attachment_plan = build_tail_attachment_plan(
                            tail_df,
                            head_delete_plan['h_clean_samples'],
                            prepare_result['grouping_embeddings_payload'],
                            args,
                            getattr(args, 'tgb_attachment_membership_mode', 'empirical_conformity'),
                        )
                        geometry = attachment_plan['geometry']
                        conformity_df = attachment_plan['conformity_df']
                        attachment_scores_df = attachment_plan['attachment_scores_df']
                        tail_open_df = attachment_plan['tail_open_df']
                        tail_attached_df = attachment_plan['tail_attached_df']
                        elbow_summary = attachment_plan['attachment_summary']
                        membership_scores_df = attachment_plan['membership_scores_df']
                        membership_calibration_df = attachment_plan['membership_calibration_df']
                        membership_summary = attachment_plan['membership_summary']
                        rgd_distances_df = attachment_plan['rgd_distances_df']
                        rgd_scores_df = attachment_plan['rgd_scores_df']
                        rgd_split_summary = attachment_plan['rgd_split_summary']
                        rgd_bic_candidates_df = attachment_plan['rgd_bic_candidates_df']
                        tail_head_normal_df, tail_head_noise_df, stable_risk_df = classify_tail_attached_stable_risk(
                            tail_attached_df,
                            args.diag_save_dir,
                            gbps_selected_iter,
                            args,
                        )
                        attachment_saved = save_tailguard_b_attachment_artifacts(
                            args.tgb_attachment_dir,
                            geometry,
                            conformity_df,
                            attachment_scores_df,
                            elbow_summary,
                            tail_open_df,
                            tail_attached_df,
                            tail_head_normal_df,
                            tail_head_noise_df,
                            membership_scores_df=membership_scores_df,
                            membership_calibration_df=membership_calibration_df,
                            membership_summary=membership_summary,
                            rgd_distances_df=rgd_distances_df,
                            rgd_scores_df=rgd_scores_df,
                            rgd_split_summary=rgd_split_summary,
                            rgd_bic_candidates_df=rgd_bic_candidates_df,
                        )
                        stable_risk_df.to_csv(os.path.join(args.tgb_attachment_dir, 'tail_attached_stable_risk.csv'), index=False)

                        stage2_plan = build_tailguard_b_stage2_plan(
                            head_delete_plan['h_clean_samples'],
                            head_delete_plan['h_removed_samples'],
                            tail_open_df,
                            tail_head_normal_df,
                            tail_head_noise_df,
                            all_samples_df=method_metadata_df,
                        )
                        stage2_saved = save_tailguard_b_stage2_artifacts(
                            args.tgb_stage2_dir,
                            stage2_plan['retained_samples'],
                            stage2_plan['removed_samples'],
                            stage2_plan['summary'],
                        )
                        denoising_diagnostics_summary, denoising_diagnostics_by_class = build_tailguard_b_denoising_diagnostics(
                            prepare_result['analysis_metadata_df'],
                            stage2_plan['retained_samples'],
                            stage2_plan['removed_samples'],
                            h_removed_samples_df=head_delete_plan['h_removed_samples'],
                            tail_head_noise_samples_df=tail_head_noise_df,
                            manifest_path=manifest_path,
                            manifest_requested_path=args.diag_manifest_path,
                            label_source=prepare_result['saved'].get('tailguard_train_analysis_metadata_csv') if prepare_result['saved'] else None,
                        )
                        denoising_diagnostics_artifacts = save_tailguard_b_denoising_diagnostics(
                            args.tgb_root_dir,
                            denoising_diagnostics_summary,
                            denoising_diagnostics_by_class,
                        )
                        cls_embeddings_payload = prepare_result['cls_embeddings_payload']
                        if cls_embeddings_payload is None:
                            raise ValueError(
                                'TailGuard-B pseudo-classification requires --tgb_save_cls_embeddings'
                            )
                        pseudoclass_registry = build_tailguard_b_pseudoclass_registry(
                            head_delete_plan['h_clean_samples'],
                            tail_head_normal_df,
                            tail_open_df,
                            stage2_plan['retained_samples'],
                            stage2_plan['removed_samples'],
                            cls_embeddings_payload,
                        )
                        pseudoclass_artifacts = save_tailguard_b_pseudoclass_artifacts(
                            os.path.join(args.tgb_root_dir, 'pseudoclasses'),
                            pseudoclass_registry['members_df'],
                            pseudoclass_registry['classes_df'],
                            pseudoclass_registry['tail_edges_df'],
                            pseudoclass_registry['summary'],
                            metadata={'dataset_provenance': args.dataset_provenance},
                        )
                        pseudoclass_report, pseudoclass_predictions, pseudoclass_summary_df, pseudoclass_contingency_df = (
                            build_tailguard_b_pseudoclass_report(
                                pseudoclass_registry['members_df'],
                                pseudoclass_registry['classes_df'],
                                cls_embeddings_payload,
                                prepare_result['analysis_metadata_df'],
                            )
                        )
                        pseudoclass_report_artifacts = save_tailguard_b_pseudoclass_report_artifacts(
                            args.tgb_root_dir,
                            pseudoclass_report,
                            pseudoclass_predictions,
                            pseudoclass_summary_df,
                            pseudoclass_contingency_df,
                        )
                        pseudoclass_status = 'completed'
                        print_fn('tailguard-b pseudo-classes: total={}, head={}, tail={}'.format(
                            pseudoclass_registry['summary']['num_pseudo_classes'],
                            pseudoclass_registry['summary']['num_head_pseudo_classes'],
                            pseudoclass_registry['summary']['num_tail_pseudo_classes'],
                        ))

                        checkpoint = load_tailguard_b_selected_checkpoint(
                            selected_checkpoint_path,
                            model,
                            optimizer=optimizer,
                            scheduler=lr_scheduler,
                            map_location=device,
                            restore_rng=True,
                        )
                        it = int(checkpoint['iteration'])
                        rebuild_result = rebuild_pruned_train_loaders(
                            base_train_data_list,
                            stage2_plan['retained_index_map'],
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
                        memory_train_eval_dataloader = train_eval_dataloader
                        gbps_has_postprocessed = True
                        reset_loader = True
                        print_fn('tailguard-b trigger iter {} selected iter {}: stage2 retained {}, removed {}'.format(
                            gbps_trigger_iter,
                            gbps_selected_iter,
                            stage2_plan['summary']['num_stage2_retained'],
                            stage2_plan['summary']['num_stage2_removed'],
                        ))
                        break
                    else:
                        gbps_has_postprocessed = True
                        print_fn('tailguard-b trigger iter {} did not run removal; continuing without Stage 2 rebuild'.format(gbps_trigger_iter))

            if current_iter % int(args.eval_interval) == 0:
                final_eval_summary = _evaluate_current_model(model, test_data_list, item_list, device, args, print_fn=print_fn)
                last_eval_iter = int(current_iter)
                model.train()

            it += 1
            if it >= total_iters:
                break

        if reset_loader:
            continue
        if len(loss_list) > 0:
            print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
        else:
            break

    if denoising_diagnostics_artifacts is None:
        if gbps_trigger_iter is None:
            cleanup_reason = 'gbps_not_triggered'
        elif args.gbps_postprocess_mode != 'remove':
            cleanup_reason = 'gbps_postprocess_mode_{}'.format(args.gbps_postprocess_mode)
        elif not os.path.isfile(selected_checkpoint_path):
            cleanup_reason = 'selected_checkpoint_unavailable'
        else:
            cleanup_reason = 'stage2_plan_unavailable'
        denoising_diagnostics_summary, denoising_diagnostics_by_class = build_tailguard_b_denoising_diagnostics(
            prepare_result['analysis_metadata_df'],
            method_metadata_df,
            method_metadata_df.iloc[0:0].copy(),
            manifest_path=manifest_path,
            manifest_requested_path=args.diag_manifest_path,
            label_source=prepare_result['saved'].get('tailguard_train_analysis_metadata_csv') if prepare_result['saved'] else None,
            cleanup_status='not_performed',
            cleanup_reason=cleanup_reason,
        )
        denoising_diagnostics_artifacts = save_tailguard_b_denoising_diagnostics(
            args.tgb_root_dir,
            denoising_diagnostics_summary,
            denoising_diagnostics_by_class,
        )

    if pseudoclass_status != 'completed':
        pseudoclass_reason = 'cleanup_status_{}'.format(
            denoising_diagnostics_summary['cleanup_status']
        )

    if last_eval_iter != int(it):
        final_eval_summary = _evaluate_current_model(model, test_data_list, item_list, device, args, print_fn=print_fn)
        last_eval_iter = int(it)
        model.train()

    checkpoint_path = save_train_checkpoint(
        args.save_dir,
        args.save_name,
        model,
        iteration=it,
        args=args,
        final_eval_summary=final_eval_summary,
    )
    print_fn('saved final model checkpoint to {} before memory evaluation'.format(checkpoint_path))

    memory_system = None
    memory_status = 'disabled' if not bool(args.tgb_memory_enable) else 'skipped_pseudoclasses_unavailable'
    if bool(args.tgb_memory_enable) and pseudoclass_registry is not None:
        memory_system = build_tailguard_b_pseudoclass_memory_system(
            model,
            memory_train_eval_dataloader,
            device,
            pseudoclass_registry['members_df'],
            pseudoclass_registry['classes_df'],
            args,
        )
        print_fn('tailguard-b memory: built {} class-wise tail banks across {} pseudo-classes'.format(
            memory_system['num_tail_memory_banks'],
            memory_system['num_pseudo_classes'],
        ))
        score_df, per_class_metrics, memory_eval_summary, metrics_by_mode = run_pseudoclass_memory_evaluation(
            model,
            test_data_list,
            item_list,
            memory_system,
            args,
            device,
        )
        memory_metadata = {
            'dataset_provenance': args.dataset_provenance,
            'checkpoint_path': args.checkpoint_path,
            'encoder_name': checkpoint_metadata['encoder_name'] if checkpoint_metadata is not None else args.encoder_name,
            'image_size': image_size,
            'crop_size': crop_size,
            'num_pseudo_classes': int(memory_system['num_pseudo_classes']),
            'num_head_pseudo_classes': int(memory_system['num_head_pseudo_classes']),
            'num_tail_pseudo_classes': int(memory_system['num_tail_pseudo_classes']),
            'num_tail_memory_banks': int(memory_system['num_tail_memory_banks']),
            'tgb_memory_fusion_lambda': float(args.tgb_memory_fusion_lambda),
            'tgb_memory_topk_ratio': float(args.tgb_memory_topk_ratio),
            'tgb_mem_max_patches_per_class': int(args.tgb_mem_max_patches_per_class),
        }
        memory_saved = save_tailguard_b_memory_artifacts(
            args.tgb_memory_dir,
            memory_system,
            score_df,
            per_class_metrics,
            memory_eval_summary,
            memory_metadata,
            metrics_by_mode=metrics_by_mode,
        )
        memory_status = 'completed'
        print_fn('saved tailguard-b memory artifacts to {}'.format(args.tgb_memory_dir))
        print_fn(json.dumps(memory_eval_summary, indent=2, ensure_ascii=False))

    total_time = time.time() - train_start_time
    time_per_iter = total_time / max(1, total_iters)
    iters_per_sec = total_iters / max(total_time, 1e-12)
    samples_per_sec = batch_size / max(time_per_iter, 1e-12)

    print_fn('Training finished. Total time: {:.2f}s ({:.2f} min), {:.4f} s/iter'.format(total_time, total_time / 60.0, time_per_iter))
    _print_eval_summary(final_eval_summary, print_fn)
    print_fn('Training efficiency:')
    print_fn('  total_time_s      {:.2f}'.format(total_time))
    print_fn('  total_iters       {}'.format(total_iters))
    print_fn('  batch_size        {}'.format(batch_size))
    print_fn('  time_per_iter_s   {:.4f}'.format(time_per_iter))
    print_fn('  iters_per_sec     {:.4f}'.format(iters_per_sec))
    print_fn('  samples_per_sec   {:.2f}'.format(samples_per_sec))

    tailguard_b_summary = {
        'dataset_provenance': args.dataset_provenance,
        'checkpoint_path': checkpoint_path,
        'input_checkpoint_path': args.checkpoint_path,
        'selected_checkpoint_path': selected_checkpoint_path if os.path.isfile(selected_checkpoint_path) else None,
        'prepare_summary': prepare_result['summary'],
        'gbps_trigger_summary': gbps_trigger_summary,
        'stage2_summary': None if stage2_plan is None else stage2_plan['summary'],
        'attachment_artifacts': attachment_saved,
        'attachment_replay_config_path': getattr(args, 'tgb_attachment_replay_config_path', None),
        'stage2_artifacts': stage2_saved,
        'denoising_diagnostics_artifacts': denoising_diagnostics_artifacts,
        'denoising_diagnostics_summary': {
            'cleanup_status': denoising_diagnostics_summary['cleanup_status'],
            'cleanup_reason': denoising_diagnostics_summary['cleanup_reason'],
            'decision_counts': denoising_diagnostics_summary['decision_counts'],
            'contamination_counts': denoising_diagnostics_summary['contamination_counts'],
            'contamination_metrics': denoising_diagnostics_summary['contamination_metrics'],
        },
        'pseudo_class_status': pseudoclass_status,
        'pseudo_class_reason': pseudoclass_reason,
        'pseudo_class_artifacts': pseudoclass_artifacts,
        'pseudo_class_report_artifacts': pseudoclass_report_artifacts,
        'pseudo_class_summary': None if pseudoclass_registry is None else {
            'num_pseudo_classes': pseudoclass_registry['summary']['num_pseudo_classes'],
            'num_head_pseudo_classes': pseudoclass_registry['summary']['num_head_pseudo_classes'],
            'num_tail_pseudo_classes': pseudoclass_registry['summary']['num_tail_pseudo_classes'],
            'num_tail_singleton_classes': pseudoclass_registry['summary']['num_tail_singleton_classes'],
        },
        'final_eval_summary': final_eval_summary,
        'memory_status': memory_status,
        'memory_config': {
            'tgb_memory_fusion_lambda': float(args.tgb_memory_fusion_lambda),
            'tgb_memory_topk_ratio': float(args.tgb_memory_topk_ratio),
            'tgb_mem_max_patches_per_class': int(args.tgb_mem_max_patches_per_class),
        },
        'memory_eval_summary': memory_eval_summary,
        'memory_artifacts': memory_saved,
        'num_pseudo_classes': 0 if memory_system is None else int(memory_system['num_pseudo_classes']),
        'num_head_pseudo_classes': 0 if memory_system is None else int(memory_system['num_head_pseudo_classes']),
        'num_tail_pseudo_classes': 0 if memory_system is None else int(memory_system['num_tail_pseudo_classes']),
        'num_tail_memory_banks': 0 if memory_system is None else int(memory_system['num_tail_memory_banks']),
        'total_time_s': float(total_time),
        'time_per_iter_s': float(time_per_iter),
        'iters_per_sec': float(iters_per_sec),
        'samples_per_sec': float(samples_per_sec),
    }
    summary_path = save_tailguard_b_summary(args.tgb_root_dir, tailguard_b_summary)
    print_fn('saved tailguard-b summary to {}'.format(summary_path))


def build_parser(default_dataset_profile='mvtec'):
    default_profile = get_dataset_profile(default_dataset_profile)
    parser = argparse.ArgumentParser(description='Standalone TailGuard-B runner for MVTec-compatible datasets')
    parser.add_argument('--dataset_profile', type=str, choices=dataset_profile_names(), default=default_profile.name)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default=None)
    parser.add_argument('--diag_save_dir', type=str, default='gbps')
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

    parser.add_argument('--tailsampler_type', type=str, default='adaptive_trim_mode', choices=['tail', 'adaptive', 'adaptive_trim_mode'])
    parser.add_argument('--tailsampler_th_type', type=str, default=None)
    parser.add_argument('--tailsampler_vote_type', type=str, default=None)
    parser.add_argument('--tailsampler_percentile', type=float, default=0.15)
    parser.add_argument('--tailsampler_embedding_source', type=str, default='encoder_cls', choices=['encoder', 'encoder_cls'])
    parser.add_argument('--tailsampler_gt_mode', type=str, default='dataset_rule', choices=['true_class', 'dataset_rule'])
    parser.add_argument('--tailsampler_tail_count_thr', type=int, default=8)
    parser.add_argument('--tailsampler_dataset_name', type=str, default=None)

    parser.add_argument('--gbps_check_start_ratio', type=float, default=0.01)
    parser.add_argument('--gbps_check_end_ratio', type=float, default=0.15)
    parser.add_argument('--gbps_check_interval', type=int, default=20)
    parser.add_argument('--gbps_grouping_method', type=str, default='kmeans')
    parser.add_argument('--gbps_num_groups', type=int, default=None)
    parser.add_argument('--gbps_auto_k', action='store_true')
    parser.add_argument('--gbps_k_candidates', type=str, default='6,8,10,12,15,20')
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
    parser.add_argument('--gbps_min_keep_per_group', type=int, default=20)
    parser.add_argument('--gbps_bootstrap_B', type=int, default=20)
    parser.add_argument('--gbps_ci_z', type=float, default=1.96)
    parser.add_argument('--gbps_improve_eps', type=float, default=1e-6)
    parser.add_argument('--gbps_min_checks_before_trigger', type=int, default=2)
    parser.add_argument('--gbps_min_checks_after_best', type=int, default=1)

    parser.add_argument('--tgb_tail_embedding_source', type=str, default='encoder_cls', choices=['encoder', 'encoder_cls'])
    parser.add_argument('--tgb_group_embedding_source', type=str, default='encoder', choices=['encoder', 'encoder_cls'])
    parser.add_argument('--tgb_stage1_training_scope', type=str, default='h_only', choices=['h_only', 'all_samples'])
    parser.add_argument('--tgb_save_cls_embeddings', dest='tgb_save_cls_embeddings', action='store_true', default=True)
    parser.add_argument('--no-tgb_save_cls_embeddings', dest='tgb_save_cls_embeddings', action='store_false')
    parser.add_argument('--tgb_save_grouping_embeddings', dest='tgb_save_grouping_embeddings', action='store_true', default=True)
    parser.add_argument('--no-tgb_save_grouping_embeddings', dest='tgb_save_grouping_embeddings', action='store_false')
    parser.add_argument('--tgb_save_tailsampler_analysis', dest='tgb_save_tailsampler_analysis', action='store_true', default=True)
    parser.add_argument('--no-tgb_save_tailsampler_analysis', dest='tgb_save_tailsampler_analysis', action='store_false')
    parser.add_argument('--tgb_save_tailsampler_analysis_details', dest='tgb_save_tailsampler_analysis_details', action='store_true', default=True)
    parser.add_argument('--no-tgb_save_tailsampler_analysis_details', dest='tgb_save_tailsampler_analysis_details', action='store_false')
    parser.add_argument('--tgb_default_adaptive_angle', type=float, default=5.0)
    parser.add_argument('--tgb_min_clean_group_size', type=int, default=3)
    parser.add_argument('--tgb_attachment_membership_mode', type=str, default='empirical_conformity', choices=['empirical_conformity', 'h_calibrated', 'rgd'])
    parser.add_argument('--tgb_elbow_min_segment', type=int, default=3)
    parser.add_argument('--tgb_rgd_split_mode', type=str, default='segmented_bic', choices=['segmented_bic', 'largest_positive_gap'])
    parser.add_argument('--tgb_stable_window', type=int, default=3)
    parser.add_argument('--tgb_stable_min_observations', type=int, default=1)
    parser.add_argument('--tgb_t_attached_noise_p_high_thr', type=float, default=0.5)
    parser.add_argument('--tgb_memory_enable', dest='tgb_memory_enable', action='store_true', default=True)
    parser.add_argument('--no-tgb_memory_enable', dest='tgb_memory_enable', action='store_false')
    parser.add_argument('--tgb_memory_fusion_lambda', type=float, default=1.0)
    parser.add_argument('--tgb_memory_topk_ratio', type=float, default=0.05)
    parser.add_argument('--tgb_mem_chunk_size', type=int, default=4096)
    parser.add_argument('--tgb_mem_max_patches_per_class', type=int, default=20000)
    return parser


def resolve_dataset_profile_args(parsed_args):
    profile = get_dataset_profile(parsed_args.dataset_profile)
    if parsed_args.data_path is None:
        parsed_args.data_path = profile.default_data_path
    if parsed_args.save_name is None:
        parsed_args.save_name = 'vitill_{}_tailguard_b'.format(profile.name)
    parsed_args.dataset_profile = profile.name
    parsed_args.dataset_item_list = list(profile.item_list)
    parsed_args.dataset_layout = profile.layout
    parsed_args.dataset_provenance = profile_provenance(profile, parsed_args.data_path)
    return profile


def main(default_dataset_profile='mvtec', argv=None, required_dataset_profile=None):
    global args, device, print_fn

    parser = build_parser(default_dataset_profile)
    args = parser.parse_args(argv)
    profile = resolve_dataset_profile_args(args)
    if required_dataset_profile is not None and profile.name != required_dataset_profile:
        parser.error('this launcher requires --dataset_profile {}'.format(required_dataset_profile))
    if not os.path.isdir(args.data_path):
        parser.error('dataset directory does not exist: {}'.format(args.data_path))
    if not args.tgb_save_cls_embeddings:
        parser.error('--tgb_save_cls_embeddings is required for TailGuard-B pseudo-class artifacts and replay')

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    args.tgb_root_dir = os.path.join(args.save_dir, args.save_name, 'tailguard_b')
    args.tgb_prepare_dir = os.path.join(args.tgb_root_dir, 'prepare')
    args.tgb_attachment_dir = os.path.join(args.tgb_root_dir, 'attachment')
    args.tgb_stage2_dir = os.path.join(args.tgb_root_dir, 'stage2')
    args.tgb_memory_dir = os.path.join(args.tgb_root_dir, 'memory')
    args.tgb_checkpoint_dir = os.path.join(args.tgb_root_dir, 'checkpoints')
    if not os.path.isabs(args.diag_save_dir):
        args.diag_save_dir = os.path.join(args.tgb_root_dir, args.diag_save_dir)
    os.makedirs(args.diag_save_dir, exist_ok=True)
    os.makedirs(args.tgb_checkpoint_dir, exist_ok=True)
    args.tgb_attachment_replay_config_path = save_tailguard_b_attachment_replay_config(
        args.tgb_root_dir,
        {
            'dataset_provenance': args.dataset_provenance,
            'tgb_attachment_membership_mode': args.tgb_attachment_membership_mode,
            'tgb_min_clean_group_size': args.tgb_min_clean_group_size,
            'tgb_elbow_min_segment': args.tgb_elbow_min_segment,
            'tgb_rgd_split_mode': args.tgb_rgd_split_mode,
            'tgb_stable_window': args.tgb_stable_window,
            'tgb_stable_min_observations': args.tgb_stable_min_observations,
            'tgb_t_attached_noise_p_high_thr': args.tgb_t_attached_noise_p_high_thr,
            'gbps_prune_ratio': args.gbps_prune_ratio,
            'gbps_min_keep_per_group': args.gbps_min_keep_per_group,
            'tgb_gbps_dir_relpath': os.path.relpath(args.diag_save_dir, args.tgb_root_dir),
            'tgb_gbps_dir': args.diag_save_dir,
        },
    )
    device = 'cuda:{}'.format(args.gpus) if torch.cuda.is_available() else 'cpu'
    print_fn(device)
    train(profile.item_list)
    return args


if __name__ == '__main__':
    main()
