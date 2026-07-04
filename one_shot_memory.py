import json
import math
import os
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F

from utils import compute_image_level_scores, get_gaussian_kernel, infer_anomaly_map_batch


MVTec_ITEM_LIST = [
    'carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
    'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper',
]


def extract_encoder_patch_tokens(model, images):
    output = model(images, return_patch_tokens=True)
    if len(output) != 4:
        raise RuntimeError('model did not return patch tokens')
    _, _, patch_tokens, spatial_size = output
    patch_tokens = F.normalize(patch_tokens, dim=-1)
    return patch_tokens, tuple(int(v) for v in spatial_size)


def should_apply_memory(class_id, class_train_counts, args):
    mode = getattr(args, 'mem_apply_mode', 'class_size_le')
    class_count = int(class_train_counts.get(int(class_id), 0))
    if mode == 'all':
        return True
    if mode == 'one_shot':
        return class_count == 1
    if mode == 'class_size_le':
        return class_count <= int(getattr(args, 'mem_class_size_thr', 8))
    raise ValueError('unsupported mem_apply_mode: {}'.format(mode))


@torch.no_grad()
def build_class_memory_bank(model, train_eval_dataloader, device, class_train_counts, args, item_list=None):
    item_list = MVTec_ITEM_LIST if item_list is None else list(item_list)
    was_training = model.training
    model.eval()
    bank = {}

    for images, _, meta in train_eval_dataloader:
        images = images.to(device)
        patch_feats, spatial_size = extract_encoder_patch_tokens(model, images)
        class_ids = meta['class_id']
        img_paths = meta['img_path']

        for index in range(images.shape[0]):
            class_id = int(class_ids[index])
            if not should_apply_memory(class_id, class_train_counts, args):
                continue

            if class_id not in bank:
                bank[class_id] = {
                    'class_id': class_id,
                    'class_name': item_list[class_id],
                    'features': [],
                    'num_images': 0,
                    'spatial_size': spatial_size,
                    'img_paths': [],
                }

            bank[class_id]['features'].append(patch_feats[index].detach().cpu())
            bank[class_id]['num_images'] += 1
            bank[class_id]['img_paths'].append(img_paths[index])

    max_patches_per_class = int(getattr(args, 'mem_max_patches_per_class', 20000))
    for class_id, entry in bank.items():
        features = torch.cat(entry['features'], dim=0)
        if max_patches_per_class > 0 and features.shape[0] > max_patches_per_class:
            keep_indices = torch.linspace(0, features.shape[0] - 1, steps=max_patches_per_class).round().long()
            features = features.index_select(0, keep_indices)
        entry['features'] = features.contiguous().cpu()
        entry['num_patches'] = int(features.shape[0])
        entry['feature_dim'] = int(features.shape[1])

    if was_training:
        model.train()
    return bank


def compute_memory_patch_scores(test_feats, memory_feats, chunk_size):
    distances = []
    memory_feats = memory_feats.to(test_feats.device)
    for start in range(0, test_feats.shape[0], chunk_size):
        end = min(start + chunk_size, test_feats.shape[0])
        query = test_feats[start:end]
        similarity = query @ memory_feats.T
        nearest_similarity = similarity.max(dim=1).values
        distances.append(1.0 - nearest_similarity)
    return torch.cat(distances, dim=0)


def pool_patch_scores(patch_scores, topk_ratio):
    patch_scores = patch_scores.reshape(1, -1)
    return compute_image_level_scores(patch_scores, max_ratio=float(topk_ratio))[0]


@torch.no_grad()
def compute_memory_scores_for_batch(model, images, class_ids, memory_bank, args, device):
    patch_feats, spatial_size = extract_encoder_patch_tokens(model, images.to(device))
    batch_scores = []
    batch_maps = []
    applied_flags = []
    chunk_size = int(getattr(args, 'mem_chunk_size', 4096))
    topk_ratio = float(getattr(args, 'mem_topk_ratio', 0.05))

    for index in range(images.shape[0]):
        class_id = int(class_ids[index])
        entry = memory_bank.get(class_id)
        if entry is None:
            batch_scores.append(torch.tensor(0.0, device=device))
            batch_maps.append(torch.zeros(spatial_size, device=device))
            applied_flags.append(0)
            continue

        patch_scores = compute_memory_patch_scores(
            patch_feats[index],
            entry['features'],
            chunk_size=chunk_size,
        )
        batch_scores.append(pool_patch_scores(patch_scores, topk_ratio=topk_ratio))
        batch_maps.append(patch_scores.reshape(spatial_size))
        applied_flags.append(1)

    return torch.stack(batch_scores), torch.stack(batch_maps), torch.tensor(applied_flags, dtype=torch.int64)


def resize_memory_map(memory_map, out_size):
    resized = F.interpolate(
        memory_map.unsqueeze(1),
        size=out_size,
        mode='bilinear',
        align_corners=False,
    )
    return resized[:, 0]


@torch.no_grad()
def run_memory_augmented_evaluation(model, test_data_list, item_list, memory_bank, args, device):
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    rows: List[dict] = []
    per_class_metrics: List[dict] = []
    score_mode = getattr(args, 'mem_score_mode', 'fused')
    lambda_value = float(getattr(args, 'mem_fusion_lambda', 1.0))
    batch_size = int(getattr(args, 'batch_size'))
    num_workers = int(getattr(args, 'num_workers'))
    max_ratio = float(getattr(args, 'max_ratio'))
    resize_mask = int(getattr(args, 'resize_mask'))

    for class_id, (class_name, test_data) in enumerate(zip(item_list, test_data_list)):
        loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        class_rows = []
        gt_sp = []
        pred_sp = []
        gt_px = []
        pred_px = []

        for images, gt, label, img_path in loader:
            images = images.to(device)
            gt = gt.to(device)
            recon_map, gt = infer_anomaly_map_batch(
                model,
                images,
                gaussian_kernel,
                resize_mask=resize_mask,
                gt=gt,
            )
            recon_scores = compute_image_level_scores(recon_map, max_ratio=max_ratio)

            memory_scores, memory_patch_maps, mem_applied = compute_memory_scores_for_batch(
                model,
                images,
                torch.as_tensor([class_id] * images.shape[0]),
                memory_bank,
                args,
                device,
            )
            memory_map = resize_memory_map(memory_patch_maps.to(device), recon_map.shape[-2:]).unsqueeze(1)

            mem_applied_mask = mem_applied.to(device=device, dtype=torch.bool)
            if score_mode == 'recon':
                final_scores = recon_scores
                final_map = recon_map
            elif score_mode == 'memory':
                final_scores = torch.where(mem_applied_mask, memory_scores, recon_scores)
                final_map = torch.where(mem_applied_mask[:, None, None, None], memory_map, recon_map)
            elif score_mode == 'fused':
                final_scores = recon_scores + lambda_value * memory_scores
                final_map = recon_map + lambda_value * memory_map
            else:
                raise ValueError('unsupported mem_score_mode: {}'.format(score_mode))

            gt_bool = gt.bool()
            if gt_bool.shape[1] > 1:
                gt_bool = torch.max(gt_bool, dim=1, keepdim=True)[0]

            gt_sp.append(label.cpu())
            pred_sp.append(final_scores.detach().cpu())
            gt_px.append(gt_bool[:, 0].detach().cpu())
            pred_px.append(final_map[:, 0].detach().cpu())

            for index in range(images.shape[0]):
                row = {
                    'class_id': class_id,
                    'class_name': class_name,
                    'img_path': img_path[index],
                    'label': int(label[index].item()),
                    'recon_score': float(recon_scores[index].item()),
                    'memory_score': float(memory_scores[index].item()),
                    'final_score': float(final_scores[index].item()),
                    'mem_applied': int(mem_applied[index].item()),
                }
                class_rows.append(row)
                rows.append(row)

        gt_sp_np = torch.cat(gt_sp).flatten().numpy()
        pred_sp_np = torch.cat(pred_sp).flatten().numpy()
        gt_px_np = torch.cat(gt_px).numpy()
        pred_px_np = torch.cat(pred_px).numpy()

        from sklearn.metrics import average_precision_score, roc_auc_score
        from utils import compute_pro, f1_score_max

        per_class_metrics.append({
            'class_id': class_id,
            'class_name': class_name,
            'I-AUROC': float(roc_auc_score(gt_sp_np, pred_sp_np)),
            'I-AP': float(average_precision_score(gt_sp_np, pred_sp_np)),
            'I-F1': float(f1_score_max(gt_sp_np, pred_sp_np)),
            'P-AUROC': float(roc_auc_score(gt_px_np.ravel(), pred_px_np.ravel())),
            'P-AP': float(average_precision_score(gt_px_np.ravel(), pred_px_np.ravel())),
            'P-F1': float(f1_score_max(gt_px_np.ravel(), pred_px_np.ravel())),
            'P-AUPRO': float(compute_pro(gt_px_np, pred_px_np)),
            'num_samples': int(len(class_rows)),
        })

    metrics_df = pd.DataFrame(per_class_metrics)
    summary = {
        'I-AUROC': float(metrics_df['I-AUROC'].mean()),
        'I-AP': float(metrics_df['I-AP'].mean()),
        'I-F1': float(metrics_df['I-F1'].mean()),
        'P-AUROC': float(metrics_df['P-AUROC'].mean()),
        'P-AP': float(metrics_df['P-AP'].mean()),
        'P-F1': float(metrics_df['P-F1'].mean()),
        'P-AUPRO': float(metrics_df['P-AUPRO'].mean()),
    }
    return pd.DataFrame(rows), per_class_metrics, summary


def save_memory_artifacts(output_dir, memory_bank, score_df, per_class_metrics, summary, metadata):
    os.makedirs(output_dir, exist_ok=True)

    memory_bank_path = os.path.join(output_dir, 'memory_bank.pt')
    torch.save(memory_bank, memory_bank_path)

    bank_summary_rows = []
    for class_id, entry in sorted(memory_bank.items()):
        bank_summary_rows.append({
            'class_id': int(entry['class_id']),
            'class_name': entry['class_name'],
            'num_images': int(entry['num_images']),
            'num_patches': int(entry['num_patches']),
            'feature_dim': int(entry['feature_dim']),
            'spatial_h': int(entry['spatial_size'][0]),
            'spatial_w': int(entry['spatial_size'][1]),
        })
    pd.DataFrame(bank_summary_rows).to_csv(os.path.join(output_dir, 'memory_bank_summary.csv'), index=False)
    score_df.to_csv(os.path.join(output_dir, 'memory_eval_scores.csv'), index=False)

    summary_payload = {
        'summary': summary,
        'per_class_metrics': per_class_metrics,
        'metadata': metadata,
    }
    with open(os.path.join(output_dir, 'memory_eval_summary.json'), 'w', encoding='utf-8') as file:
        json.dump(summary_payload, file, indent=2, ensure_ascii=False)

    return {
        'memory_bank_path': memory_bank_path,
        'memory_bank_summary_path': os.path.join(output_dir, 'memory_bank_summary.csv'),
        'memory_eval_scores_path': os.path.join(output_dir, 'memory_eval_scores.csv'),
        'memory_eval_summary_path': os.path.join(output_dir, 'memory_eval_summary.json'),
    }
