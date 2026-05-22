import argparse
import logging
import os
import random
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, Subset
from torchvision.datasets import ImageFolder

from dataset import MVTecDataset, TrainDiagDataset, TrainWeightDataset, get_data_transforms
from dinov1.utils import trunc_normal_
from models import vit_encoder
from models.uad import ViTill
from models.vision_transformer import Block as VitBlock, LinearAttention2, bMlp
from optimizers import StableAdamW
from utils import WarmCosineScheduler, evaluation_batch, global_cosine_hm_percent
from warmup_diag import (
    build_phase2_weight_update,
    build_warmup_postprocess_plan,
    evaluate_warmup_trigger,
    load_injected_manifest,
    load_saved_train_scores,
    parse_warmup_milestones,
    run_one_warmup_diagnosis,
    save_phase2_artifacts,
    save_prune_plan,
)

warnings = __import__('warnings')
warnings.filterwarnings("ignore")


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_format)
    logger.addHandler(stream_handler)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_train_dataloader(train_datasets, batch_size, num_workers, data_root=None, item_list=None, with_meta=False):
    if with_meta:
        wrapped_datasets = []
        for class_id, (item, train_data) in enumerate(zip(item_list, train_datasets)):
            wrapped_datasets.append(
                TrainWeightDataset(
                    train_data,
                    data_root=data_root,
                    class_name=item,
                    class_id=class_id,
                )
            )
        train_data = ConcatDataset(wrapped_datasets)
    else:
        train_data = ConcatDataset(train_datasets)

    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    return train_data, train_dataloader


def build_train_eval_dataloader(train_datasets, data_root, contaminated_paths, batch_size, num_workers, item_list):
    train_eval_data_list = []
    sample_offset = 0
    for class_id, (item, train_data) in enumerate(zip(item_list, train_datasets)):
        train_eval_data = TrainDiagDataset(
            train_data,
            data_root=data_root,
            class_name=item,
            class_id=class_id,
            sample_offset=sample_offset,
            contaminated_paths=contaminated_paths,
        )
        train_eval_data_list.append(train_eval_data)
        sample_offset += len(train_eval_data)

    train_eval_data = ConcatDataset(train_eval_data_list)
    train_eval_dataloader = torch.utils.data.DataLoader(
        train_eval_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_eval_dataloader


def resolve_postprocess_mode(args):
    if args.warmup_denoise and args.warmup_postprocess_mode == 'none':
        return 'remove'
    return args.warmup_postprocess_mode


def build_batch_sample_weight(meta, sample_weight_map, device):
    weights = []
    for class_id, base_idx in zip(meta['class_id'], meta['base_idx']):
        if torch.is_tensor(class_id):
            class_id = class_id.item()
        if torch.is_tensor(base_idx):
            base_idx = base_idx.item()
        weights.append(float(sample_weight_map.get((int(class_id), int(base_idx)), 1.0)))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def should_run_warmup_check(current_iter, total_iters, args):
    if not args.warmup_auto_end:
        return False
    t_min = max(1, int(total_iters * args.warmup_check_start_ratio))
    t_max = max(t_min, int(total_iters * args.warmup_check_end_ratio))
    if current_iter < t_min or current_iter > t_max:
        return False
    return (current_iter - t_min) % args.warmup_check_interval == 0 or current_iter == t_max


def run_phase2_diagnosis(model, train_eval_dataloader, current_iter, manifest_path):
    return run_one_warmup_diagnosis(
        model,
        train_eval_dataloader,
        device,
        save_dir=args.diag_save_dir,
        current_iter=current_iter,
        print_fn=print_fn,
        max_ratio=args.diag_max_ratio,
        resize_mask=args.diag_resize_mask,
        manifest_path=manifest_path,
        save_scores=True,
    )


def train(item_list):
    setup_seed(1)

    total_iters = 10000
    batch_size = 16
    num_workers = 4
    image_size = 448
    crop_size = 392
    train_start_time = time.time()
    final_eval_summary = None
    postprocess_mode = resolve_postprocess_mode(args)
    phase2_enabled = args.warmup_auto_end
    use_sample_weight = phase2_enabled or postprocess_mode in ['soft', 'hybrid']

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_data_list = []
    test_data_list = []
    contaminated_paths, manifest_path = (None, None)
    if args.warmup_diag or postprocess_mode != 'none' or phase2_enabled:
        contaminated_paths, manifest_path = load_injected_manifest(
            args.data_path,
            os.path.dirname(__file__),
            manifest_path=args.diag_manifest_path,
        )
        if contaminated_paths is None:
            print_fn('warmup diagnosis: inject_defects manifest not found, contamination-aware metrics will be skipped.')
        else:
            print_fn('warmup diagnosis: loaded contamination manifest {}.'.format(manifest_path))

    if phase2_enabled and args.warmup_primary_metric == 'score_gap' and contaminated_paths is None:
        raise ValueError('--warmup_primary_metric score_gap requires contamination-aware metrics. Please provide a valid inject_defects manifest via --diag_manifest_path.')

    for i, item in enumerate(item_list):
        train_path = os.path.join(args.data_path, item, 'train')
        test_path = os.path.join(args.data_path, item)

        train_data = ImageFolder(root=train_path, transform=data_transform)
        train_data.classes = item
        train_data.class_to_idx = {item: i}
        train_data.samples = [(sample[0], i) for sample in train_data.samples]

        test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
        train_data_list.append(train_data)
        test_data_list.append(test_data)

    train_data, train_dataloader = build_train_dataloader(
        train_data_list,
        batch_size=batch_size,
        num_workers=num_workers,
        data_root=args.data_path,
        item_list=item_list,
        with_meta=use_sample_weight,
    )

    train_eval_dataloader = None
    if args.warmup_diag or postprocess_mode != 'none' or phase2_enabled:
        train_eval_dataloader = build_train_eval_dataloader(
            train_data_list,
            data_root=args.data_path,
            contaminated_paths=contaminated_paths,
            batch_size=args.diag_batch_size,
            num_workers=args.diag_num_workers,
            item_list=item_list,
        )

    encoder_name = 'dinov2reg_vit_base_14'
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(encoder_name)

    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise RuntimeError('Architecture not in small, base, large.')

    bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])

    decoder = []
    for _ in range(8):
        blk = VitBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-8),
            attn=LinearAttention2,
        )
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    model = ViTill(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        mask_neighbor_size=0,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
    )
    model = model.to(device)
    trainable = nn.ModuleList([bottleneck, decoder])

    for module in trainable.modules():
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.01, a=-0.03, b=0.03)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    optimizer = StableAdamW(
        [{'params': trainable.parameters()}],
        lr=2e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        amsgrad=True,
        eps=1e-10,
    )
    lr_scheduler = WarmCosineScheduler(
        optimizer,
        base_value=2e-3,
        final_value=2e-4,
        total_iters=total_iters,
        warmup_iters=100,
    )

    print_fn('train image number:{}'.format(len(train_data)))

    has_postprocessed = False
    sample_weight_map = {}
    reliability_map = {}
    stage = 'warmup' if phase2_enabled else 'baseline'
    warmup_end_iter = None
    warmup_trigger_iter = None
    freeze_iter = None
    warmup_trigger_reason = None
    best_primary_value = None
    best_primary_iter = None
    no_improve_count = 0
    last_suspicious_keys = None
    last_refresh_iter = None

    it = 0
    for _ in range(int(np.ceil(total_iters / len(train_dataloader)))):
        model.train()
        loss_list = []

        for batch in train_dataloader:
            if use_sample_weight:
                img, label, meta = batch
                batch_sample_weight = build_batch_sample_weight(meta, sample_weight_map, device)
            else:
                img, label = batch
                meta = None
                batch_sample_weight = None

            img = img.to(device)
            label = label.to(device)

            en, de = model(img)

            p_final = 0.9
            p = min(p_final * it / 1000, p_final)
            loss = global_cosine_hm_percent(en, de, p=p, factor=0.1, sample_weight=batch_sample_weight)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)

            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()

            current_iter = it + 1
            if args.warmup_diag and current_iter in args.warmup_milestones and not (
                (postprocess_mode != 'none' and current_iter == args.warmup_end_iter) or
                (phase2_enabled and stage == 'warmup' and should_run_warmup_check(current_iter, total_iters, args))
            ):
                run_one_warmup_diagnosis(
                    model,
                    train_eval_dataloader,
                    device,
                    save_dir=args.diag_save_dir,
                    current_iter=current_iter,
                    print_fn=print_fn,
                    max_ratio=args.diag_max_ratio,
                    resize_mask=args.diag_resize_mask,
                    manifest_path=manifest_path,
                )

            if phase2_enabled and stage == 'warmup' and should_run_warmup_check(current_iter, total_iters, args):
                diagnosis_result = run_phase2_diagnosis(model, train_eval_dataloader, current_iter, manifest_path)
                t_max = max(1, int(total_iters * args.warmup_check_end_ratio))
                trigger_result = evaluate_warmup_trigger(
                    diagnosis_result['df'].copy(),
                    best_primary_value=best_primary_value,
                    best_primary_iter=best_primary_iter,
                    no_improve_count=no_improve_count,
                    last_suspicious_keys=last_suspicious_keys,
                    current_iter=current_iter,
                    top_p=args.warmup_gmm_top_p,
                    jaccard_threshold=args.warmup_jaccard_threshold,
                    patience=args.warmup_gmm_patience,
                    plateau_ratio=args.warmup_gmm_plateau_ratio,
                    primary_metric_name=args.warmup_primary_metric,
                    trigger_mode=args.warmup_trigger_mode,
                    force_trigger=current_iter >= t_max,
                )
                save_phase2_artifacts(
                    diagnosis_result['iter_dir'],
                    extra_summary=trigger_result['summary'],
                    suspicious_df=trigger_result['suspicious_df'],
                )
                best_primary_value = trigger_result['best_primary_value']
                best_primary_iter = trigger_result['best_primary_iter']
                no_improve_count = trigger_result['no_improve_count']
                last_suspicious_keys = trigger_result['suspicious_keys']
                if trigger_result['triggered']:
                    warmup_trigger_iter = current_iter
                    warmup_end_iter = trigger_result['best_primary_iter']
                    if warmup_end_iter is None:
                        raise ValueError('warm-up trigger fired without a best_primary_iter')

                    if warmup_end_iter == current_iter:
                        weight_scored_df = trigger_result['scored_df']
                        weight_source_iter = current_iter
                    else:
                        weight_scored_df, _ = load_saved_train_scores(args.diag_save_dir, warmup_end_iter)
                        weight_source_iter = warmup_end_iter

                    weight_update = build_phase2_weight_update(
                        weight_scored_df,
                        reliability_map=reliability_map,
                        old_weight_map=sample_weight_map,
                        min_weight=args.denoise_min_weight,
                        weight_beta=args.warmup_weight_beta,
                        weight_scope=args.denoise_weight_scope,
                        reliability_ema=args.denoise_reliability_ema,
                        weight_ema=args.denoise_weight_ema,
                        clip_delta=args.denoise_weight_clip_delta,
                        top_p=args.warmup_gmm_top_p,
                    )
                    freeze_iter = max(warmup_end_iter, int(total_iters * args.denoise_freeze_ratio))
                    save_phase2_artifacts(
                        diagnosis_result['iter_dir'],
                        extra_summary={
                            'stage': 'denoise',
                            'warmup_trigger_iter': int(warmup_trigger_iter),
                            'warmup_end_iter': int(warmup_end_iter),
                            'weight_init_source_iter': int(weight_source_iter),
                            'best_iter_less_than_trigger_iter': bool(warmup_end_iter < warmup_trigger_iter),
                            'warmup_trigger_reason': trigger_result['trigger_reason'],
                            'freeze_iter': int(freeze_iter),
                            **weight_update['summary'],
                        },
                        suspicious_df=weight_update['suspicious_df'],
                        weight_df=weight_update['weight_df'],
                        reliability_df=weight_update['reliability_df'],
                    )
                    sample_weight_map = weight_update['sample_weight_map']
                    reliability_map = weight_update['reliability_map']
                    stage = 'denoise'
                    warmup_trigger_reason = trigger_result['trigger_reason']
                    last_refresh_iter = current_iter
                    print_fn('phase2 warmup end iter {} (trigger iter {}): metric={}, mode={}, reason={}, freeze_iter={}, weight_mean={}, weight_min={}, weight_max={}'.format(
                        warmup_end_iter,
                        warmup_trigger_iter,
                        args.warmup_primary_metric,
                        args.warmup_trigger_mode,
                        warmup_trigger_reason,
                        freeze_iter,
                        weight_update['summary'].get('weight_mean'),
                        weight_update['summary'].get('weight_min'),
                        weight_update['summary'].get('weight_max'),
                    ))

            if phase2_enabled and stage == 'denoise' and current_iter > warmup_end_iter:
                if current_iter >= freeze_iter:
                    diagnosis_result = run_phase2_diagnosis(model, train_eval_dataloader, current_iter, manifest_path)
                    save_phase2_artifacts(
                        diagnosis_result['iter_dir'],
                        extra_summary={
                            'stage': 'freeze',
                            'warmup_end_iter': int(warmup_end_iter),
                            'freeze_iter': int(freeze_iter),
                            'warmup_trigger_reason': warmup_trigger_reason,
                            'weight_mean': float(np.mean(list(sample_weight_map.values()))) if len(sample_weight_map) > 0 else None,
                            'weight_min': float(np.min(list(sample_weight_map.values()))) if len(sample_weight_map) > 0 else None,
                            'weight_max': float(np.max(list(sample_weight_map.values()))) if len(sample_weight_map) > 0 else None,
                            'reliability_mean': float(np.mean(list(reliability_map.values()))) if len(reliability_map) > 0 else None,
                            'reliability_min': float(np.min(list(reliability_map.values()))) if len(reliability_map) > 0 else None,
                            'reliability_max': float(np.max(list(reliability_map.values()))) if len(reliability_map) > 0 else None,
                        },
                    )
                    stage = 'freeze'
                    print_fn('phase2 freeze iter {}: weights frozen.'.format(current_iter))
                elif last_refresh_iter is None or (current_iter - last_refresh_iter) >= args.denoise_refresh_interval:
                    diagnosis_result = run_phase2_diagnosis(model, train_eval_dataloader, current_iter, manifest_path)
                    weight_update = build_phase2_weight_update(
                        diagnosis_result['df'].copy(),
                        reliability_map=reliability_map,
                        old_weight_map=sample_weight_map,
                        min_weight=args.denoise_min_weight,
                        weight_beta=args.warmup_weight_beta,
                        weight_scope=args.denoise_weight_scope,
                        reliability_ema=args.denoise_reliability_ema,
                        weight_ema=args.denoise_weight_ema,
                        clip_delta=args.denoise_weight_clip_delta,
                        top_p=args.warmup_gmm_top_p,
                    )
                    save_phase2_artifacts(
                        diagnosis_result['iter_dir'],
                        extra_summary={
                            'stage': 'denoise',
                            'warmup_end_iter': int(warmup_end_iter),
                            'freeze_iter': int(freeze_iter),
                            'last_refresh_iter': int(current_iter),
                            **weight_update['summary'],
                        },
                        suspicious_df=weight_update['suspicious_df'],
                        weight_df=weight_update['weight_df'],
                        reliability_df=weight_update['reliability_df'],
                    )
                    sample_weight_map = weight_update['sample_weight_map']
                    reliability_map = weight_update['reliability_map']
                    last_refresh_iter = current_iter
                    print_fn('phase2 refresh iter {}: weight_mean={}, delta_mean_abs={}, clipped={}'.format(
                        current_iter,
                        weight_update['summary'].get('weight_mean'),
                        weight_update['summary'].get('weight_delta_mean_abs'),
                        weight_update['summary'].get('clipped_sample_count'),
                    ))

            if (not phase2_enabled) and postprocess_mode != 'none' and (not has_postprocessed) and current_iter == args.warmup_end_iter:
                diagnosis_result = run_one_warmup_diagnosis(
                    model,
                    train_eval_dataloader,
                    device,
                    save_dir=args.diag_save_dir,
                    current_iter=current_iter,
                    print_fn=print_fn,
                    max_ratio=args.diag_max_ratio,
                    resize_mask=args.diag_resize_mask,
                    manifest_path=manifest_path,
                    save_scores=args.warmup_save_scores_before_prune or postprocess_mode in ['soft', 'hybrid'],
                )
                scored_df = diagnosis_result['df'].copy()

                postprocess_prune_ratio = args.warmup_hybrid_prune_ratio if postprocess_mode == 'hybrid' else args.warmup_prune_ratio
                postprocess_plan = build_warmup_postprocess_plan(
                    scored_df,
                    mode=postprocess_mode,
                    prune_ratio=postprocess_prune_ratio,
                    min_keep_per_class=args.warmup_min_keep_per_class,
                    weight_beta=args.warmup_weight_beta,
                    min_weight=args.warmup_min_weight,
                    weight_scope=args.warmup_weight_scope,
                )
                save_prune_plan(diagnosis_result['iter_dir'], postprocess_plan, current_iter)

                if postprocess_mode in ['remove', 'hybrid']:
                    pruned_train_data_list = []
                    for class_id, train_dataset in enumerate(train_data_list):
                        retained_indices = postprocess_plan['retained_index_map'].get(class_id)
                        if retained_indices is None:
                            retained_indices = list(range(len(train_dataset)))
                        pruned_train_data_list.append(Subset(train_dataset, retained_indices))
                    train_data_list = pruned_train_data_list

                sample_weight_map = postprocess_plan['sample_weight_map']
                use_sample_weight = postprocess_mode in ['soft', 'hybrid']
                train_data, train_dataloader = build_train_dataloader(
                    train_data_list,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    data_root=args.data_path,
                    item_list=item_list,
                    with_meta=use_sample_weight,
                )
                train_eval_dataloader = build_train_eval_dataloader(
                    train_data_list,
                    data_root=args.data_path,
                    contaminated_paths=contaminated_paths,
                    batch_size=args.diag_batch_size,
                    num_workers=args.diag_num_workers,
                    item_list=item_list,
                )
                has_postprocessed = True
                print_fn('warmup postprocess iter {}: mode={}, pruned {} / {} training samples.'.format(
                    current_iter,
                    postprocess_mode,
                    postprocess_plan['summary'].get('num_pruned', 0),
                    postprocess_plan['summary'].get('num_samples_before_prune', len(scored_df)),
                ))
                if 'weight_mean' in postprocess_plan['summary']:
                    print_fn('warmup weights: mean={}, min={}, max={}'.format(
                        postprocess_plan['summary'].get('weight_mean'),
                        postprocess_plan['summary'].get('weight_min'),
                        postprocess_plan['summary'].get('weight_max'),
                    ))
                print_fn('train image number after postprocess:{}'.format(len(train_data)))
                model.train()

            if current_iter % 5000 == 0:
                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                for item, test_data in zip(item_list, test_data_list):
                    test_dataloader = torch.utils.data.DataLoader(
                        test_data,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=4,
                    )
                    results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

                    auroc_sp_list.append(auroc_sp)
                    ap_sp_list.append(ap_sp)
                    f1_sp_list.append(f1_sp)
                    auroc_px_list.append(auroc_px)
                    ap_px_list.append(ap_px)
                    f1_px_list.append(f1_px)
                    aupro_px_list.append(aupro_px)

                    print_fn(
                        '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                            item,
                            auroc_sp,
                            ap_sp,
                            f1_sp,
                            auroc_px,
                            ap_px,
                            f1_px,
                            aupro_px,
                        )
                    )

                mean_auroc_sp = np.mean(auroc_sp_list)
                mean_ap_sp = np.mean(ap_sp_list)
                mean_f1_sp = np.mean(f1_sp_list)
                mean_auroc_px = np.mean(auroc_px_list)
                mean_ap_px = np.mean(ap_px_list)
                mean_f1_px = np.mean(f1_px_list)
                mean_aupro_px = np.mean(aupro_px_list)

                print_fn(
                    'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                        mean_auroc_sp,
                        mean_ap_sp,
                        mean_f1_sp,
                        mean_auroc_px,
                        mean_ap_px,
                        mean_f1_px,
                        mean_aupro_px,
                    )
                )
                final_eval_summary = {
                    'I-AUROC': mean_auroc_sp,
                    'I-AP': mean_ap_sp,
                    'I-F1': mean_f1_sp,
                    'P-AUROC': mean_auroc_px,
                    'P-AP': mean_ap_px,
                    'P-F1': mean_f1_px,
                    'P-AUPRO': mean_aupro_px,
                }
                model.train()

            it += 1
            if it == total_iters:
                break
        print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))

    total_time = time.time() - train_start_time
    time_per_iter = total_time / total_iters
    iters_per_sec = total_iters / total_time
    samples_per_sec = batch_size / time_per_iter

    print_fn('Training finished. Total time: {:.2f}s ({:.2f} min), {:.4f} s/iter'.format(
        total_time,
        total_time / 60.0,
        time_per_iter,
    ))
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
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='../mvtec_anomaly_detection')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='vitill_mvtec_uni_dinov2br_c392_en29_bn4dp2_de8_laelu_md2_i1_it10k_sams2e3_wd1e4_w1hcosa2e4_ghmp09f01w01_b16_s1')
    parser.add_argument('--warmup_diag', action='store_true')
    parser.add_argument('--warmup_denoise', action='store_true')
    parser.add_argument('--warmup_postprocess_mode', type=str, default='none', choices=['none', 'remove', 'soft', 'hybrid'])
    parser.add_argument('--warmup_auto_end', action='store_true')
    parser.add_argument('--warmup_milestones', type=str, default='50, 100, 200, 400, 600,1000,2000,4000')
    parser.add_argument('--warmup_end_iter', type=int, default=1000)
    parser.add_argument('--warmup_prune_ratio', type=float, default=0.05)
    parser.add_argument('--warmup_hybrid_prune_ratio', type=float, default=0.05)
    parser.add_argument('--warmup_min_keep_per_class', type=int, default=20)
    parser.add_argument('--warmup_weight_beta', type=float, default=1.0)
    parser.add_argument('--warmup_min_weight', type=float, default=0.2)
    parser.add_argument('--warmup_weight_scope', type=str, default='class', choices=['class', 'global'])
    parser.add_argument('--warmup_save_scores_before_prune', action='store_true')
    parser.add_argument('--warmup_check_start_ratio', type=float, default=0.01)
    parser.add_argument('--warmup_check_end_ratio', type=float, default=0.15)
    parser.add_argument('--warmup_check_interval', type=int, default=20)
    parser.add_argument('--warmup_gmm_top_p', type=float, default=0.1)
    parser.add_argument('--warmup_primary_metric', type=str, default='D_t', choices=['D_t', 'score_gap'])
    parser.add_argument('--warmup_trigger_mode', type=str, default='plateau', choices=['plateau', 'peak_patience'])
    parser.add_argument('--warmup_jaccard_threshold', type=float, default=0.7)
    parser.add_argument('--warmup_gmm_patience', type=int, default=2)
    parser.add_argument('--warmup_gmm_plateau_ratio', type=float, default=0.95)
    parser.add_argument('--denoise_refresh_interval', type=int, default=500)
    parser.add_argument('--denoise_reliability_ema', type=float, default=0.9)
    parser.add_argument('--denoise_weight_ema', type=float, default=0.8)
    parser.add_argument('--denoise_weight_clip_delta', type=float, default=0.1)
    parser.add_argument('--denoise_min_weight', type=float, default=0.2)
    parser.add_argument('--denoise_weight_scope', type=str, default='class', choices=['class', 'global'])
    parser.add_argument('--denoise_freeze_ratio', type=float, default=0.7)
    parser.add_argument('--diag_save_dir', type=str, default='./warmup_diag')
    parser.add_argument('--diag_manifest_path', type=str, default=None)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--diag_batch_size', type=int, default=16)
    parser.add_argument('--diag_num_workers', type=int, default=4)
    parser.add_argument('--diag_max_ratio', type=float, default=0.01)
    parser.add_argument('--diag_resize_mask', type=int, default=256)
    args = parser.parse_args()
    args.warmup_milestones = parse_warmup_milestones(args.warmup_milestones)

    item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
                 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    if not os.path.isabs(args.diag_save_dir):
        args.diag_save_dir = os.path.join(args.save_dir, args.save_name, args.diag_save_dir)

    device = 'cuda:{}'.format(args.gpus) if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    train(item_list)
