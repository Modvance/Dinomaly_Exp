from typing import Dict, List

import pandas as pd
import torch
import torch.nn.functional as F

from one_shot_memory import (
    compute_memory_patch_scores,
    extract_encoder_patch_tokens,
    pool_patch_scores,
    resize_memory_map,
)
from utils import compute_image_level_scores, get_gaussian_kernel, infer_anomaly_map_batch


def _normalize_rows(tensor: torch.Tensor) -> torch.Tensor:
    return F.normalize(tensor.float(), dim=-1)


def _extract_encoder_cls_embeddings(model, images: torch.Tensor) -> torch.Tensor:
    encoder = getattr(model, 'encoder', None)
    if encoder is None:
        raise ValueError('model does not expose encoder for cls routing')
    if not hasattr(encoder, 'prepare_tokens_with_masks'):
        raise ValueError('TailGuard-B memory routing requires a DINOv2-style encoder')
    x = encoder.prepare_tokens_with_masks(images)
    for blk in encoder.blocks:
        x = blk(x)
    x = encoder.norm(x)
    return _normalize_rows(x[:, 0])


def _build_support_routing_cache(memory_bank: Dict[int, Dict], device):
    if len(memory_bank) == 0:
        return None
    support_ids = sorted(memory_bank.keys())
    support_cls = torch.stack([memory_bank[support_id]['support_cls'] for support_id in support_ids], dim=0).to(device)
    support_keys = torch.tensor([int(memory_bank[support_id].get('sample_key', support_id)) for support_id in support_ids], dtype=torch.int64)
    adaptive_angles = torch.tensor([float(memory_bank[support_id]['adaptive_angle']) for support_id in support_ids], dtype=torch.float32)
    return {
        'support_ids': support_ids,
        'support_cls': _normalize_rows(support_cls),
        'support_keys': support_keys,
        'adaptive_angles': adaptive_angles,
    }


@torch.no_grad()
def build_tail_support_memory_bank(model,
                                   train_eval_dataloader,
                                   device,
                                   tail_open_df: pd.DataFrame,
                                   args) -> Dict[int, Dict]:
    support_df = tail_open_df.copy()
    if len(support_df) == 0:
        return {}
    if 'img_path' not in support_df.columns or 'sample_idx' not in support_df.columns:
        raise ValueError('TailGuard-B memory support table must contain img_path and sample_idx')

    support_df = support_df.drop_duplicates(subset=['sample_idx']).reset_index(drop=True)
    support_lookup = {}
    for _, row in support_df.iterrows():
        sample_idx = int(row['sample_idx'])
        support_lookup[str(row['img_path'])] = {
            'sample_idx': sample_idx,
            'sample_key': int(row.get('sample_key', sample_idx)),
            'img_path': str(row['img_path']),
            'adaptive_angle': float(row.get('adaptive_angle', getattr(args, 'tgb_default_adaptive_angle', 5.0))),
        }

    was_training = model.training
    model.eval()
    bank = {}
    max_patches_per_support = int(getattr(args, 'tgb_mem_max_patches_per_support', getattr(args, 'mem_max_patches_per_class', 20000)))

    try:
        for images, _, meta in train_eval_dataloader:
            images = images.to(device)
            patch_feats, spatial_size = extract_encoder_patch_tokens(model, images)
            cls_embeddings = _extract_encoder_cls_embeddings(model, images)
            img_paths = meta['img_path']

            for index in range(images.shape[0]):
                img_path = str(img_paths[index])
                support_info = support_lookup.get(img_path)
                if support_info is None:
                    continue
                sample_idx = int(support_info['sample_idx'])
                features = patch_feats[index].detach().cpu().contiguous()
                if max_patches_per_support > 0 and features.shape[0] > max_patches_per_support:
                    keep_indices = torch.linspace(0, features.shape[0] - 1, steps=max_patches_per_support).round().long()
                    features = features.index_select(0, keep_indices)
                bank[sample_idx] = {
                    'sample_idx': sample_idx,
                    'sample_key': int(support_info['sample_key']),
                    'img_path': img_path,
                    'support_cls': cls_embeddings[index].detach().cpu().contiguous(),
                    'adaptive_angle': float(support_info['adaptive_angle']),
                    'features': features,
                    'num_patches': int(features.shape[0]),
                    'feature_dim': int(features.shape[1]),
                    'spatial_size': tuple(int(value) for value in spatial_size),
                }
    finally:
        if was_training:
            model.train()
    return bank


def route_to_tail_support(test_cls: torch.Tensor, memory_bank: Dict[int, Dict], routing_cache=None) -> Dict:
    if len(memory_bank) == 0:
        return {
            'mem_applied': 0,
            'routed_sample_idx': -1,
            'routed_sample_key': -1,
            'route_similarity': 0.0,
            'route_angle': 180.0,
            'adaptive_angle': 0.0,
        }

    if routing_cache is None:
        routing_cache = _build_support_routing_cache(memory_bank, test_cls.device)
    support_ids = routing_cache['support_ids']
    similarity = routing_cache['support_cls'] @ test_cls
    best_index = int(torch.argmax(similarity).item())
    best_sample_idx = int(support_ids[best_index])
    best_similarity = float(similarity[best_index].item())
    best_angle = float(torch.rad2deg(torch.acos(torch.clamp(similarity[best_index], min=-1.0, max=1.0))).item())
    adaptive_angle = float(routing_cache['adaptive_angles'][best_index].item())
    mem_applied = int(best_angle <= adaptive_angle)
    return {
        'mem_applied': mem_applied,
        'routed_sample_idx': best_sample_idx,
        'routed_sample_key': int(routing_cache['support_keys'][best_index].item()),
        'route_similarity': best_similarity,
        'route_angle': best_angle,
        'adaptive_angle': adaptive_angle,
    }


@torch.no_grad()
def compute_tail_support_memory_scores_for_batch(model,
                                                 images: torch.Tensor,
                                                 memory_bank: Dict[int, Dict],
                                                 args,
                                                 device):
    patch_feats, spatial_size = extract_encoder_patch_tokens(model, images.to(device))
    cls_embeddings = _extract_encoder_cls_embeddings(model, images.to(device))
    chunk_size = int(getattr(args, 'tgb_mem_chunk_size', getattr(args, 'mem_chunk_size', 4096)))
    topk_ratio = float(getattr(args, 'tgb_memory_topk_ratio', getattr(args, 'mem_topk_ratio', 0.05)))

    scores = []
    maps = []
    applied_flags = []
    routed_sample_indices = []
    routed_sample_keys = []
    route_similarities = []
    route_angles = []
    adaptive_angles = []
    routing_cache = _build_support_routing_cache(memory_bank, device)

    for index in range(images.shape[0]):
        route = route_to_tail_support(cls_embeddings[index], memory_bank, routing_cache=routing_cache)
        applied_flags.append(int(route['mem_applied']))
        routed_sample_indices.append(int(route['routed_sample_idx']))
        routed_sample_keys.append(int(route['routed_sample_key']))
        route_similarities.append(float(route['route_similarity']))
        route_angles.append(float(route['route_angle']))
        adaptive_angles.append(float(route['adaptive_angle']))

        if route['mem_applied'] == 0:
            scores.append(torch.tensor(0.0, device=device))
            maps.append(torch.zeros(spatial_size, device=device))
            continue

        entry = memory_bank[int(route['routed_sample_idx'])]
        patch_scores = compute_memory_patch_scores(
            patch_feats[index],
            entry['features'],
            chunk_size=chunk_size,
        )
        scores.append(pool_patch_scores(patch_scores, topk_ratio=topk_ratio))
        maps.append(patch_scores.reshape(spatial_size))

    return {
        'memory_scores': torch.stack(scores),
        'memory_patch_maps': torch.stack(maps),
        'mem_applied': torch.tensor(applied_flags, dtype=torch.int64),
        'routed_sample_idx': torch.tensor(routed_sample_indices, dtype=torch.int64),
        'routed_sample_key': torch.tensor(routed_sample_keys, dtype=torch.int64),
        'route_similarity': torch.tensor(route_similarities, dtype=torch.float32),
        'route_angle': torch.tensor(route_angles, dtype=torch.float32),
        'adaptive_angle': torch.tensor(adaptive_angles, dtype=torch.float32),
    }


@torch.no_grad()
def run_tail_support_memory_evaluation(model,
                                       test_data_list,
                                       item_list,
                                       memory_bank: Dict[int, Dict],
                                       args,
                                       device):
    was_training = model.training
    model.eval()
    try:
        gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
        rows: List[dict] = []
        lambda_value = float(getattr(args, 'tgb_memory_fusion_lambda', 1.0))
        batch_size = int(getattr(args, 'batch_size'))
        num_workers = int(getattr(args, 'num_workers'))
        max_ratio = float(getattr(args, 'diag_max_ratio', getattr(args, 'max_ratio', 0.01)))
        resize_mask = int(getattr(args, 'diag_resize_mask', getattr(args, 'resize_mask', 256)))
        metric_modes = ('recon', 'memory', 'fused')
        per_class_metrics_by_mode = {mode: [] for mode in metric_modes}

        for class_id, (class_name, test_data) in enumerate(zip(item_list, test_data_list)):
            loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            class_predictions = {
                mode: {'gt_sp': [], 'pred_sp': [], 'gt_px': [], 'pred_px': []}
                for mode in metric_modes
            }
            class_row_count = 0

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
                memory_result = compute_tail_support_memory_scores_for_batch(model, images, memory_bank, args, device)
                memory_scores = memory_result['memory_scores']
                memory_patch_maps = memory_result['memory_patch_maps']
                memory_map = resize_memory_map(memory_patch_maps.to(device), recon_map.shape[-2:]).unsqueeze(1)
                mem_applied_mask = memory_result['mem_applied'].to(device=device, dtype=torch.bool)

                fused_scores = torch.where(mem_applied_mask, recon_scores + lambda_value * memory_scores, recon_scores)
                fused_maps = torch.where(mem_applied_mask[:, None, None, None], recon_map + lambda_value * memory_map, recon_map)
                memory_mode_scores = torch.where(mem_applied_mask, memory_scores, recon_scores)
                memory_mode_maps = torch.where(mem_applied_mask[:, None, None, None], memory_map, recon_map)
                scores_by_mode = {'recon': recon_scores, 'memory': memory_mode_scores, 'fused': fused_scores}
                maps_by_mode = {'recon': recon_map, 'memory': memory_mode_maps, 'fused': fused_maps}

                gt_bool = gt.bool()
                if gt_bool.shape[1] > 1:
                    gt_bool = torch.max(gt_bool, dim=1, keepdim=True)[0]

                for mode in metric_modes:
                    class_predictions[mode]['gt_sp'].append(label.cpu())
                    class_predictions[mode]['pred_sp'].append(scores_by_mode[mode].detach().cpu())
                    class_predictions[mode]['gt_px'].append(gt_bool[:, 0].detach().cpu())
                    class_predictions[mode]['pred_px'].append(maps_by_mode[mode][:, 0].detach().cpu())

                for index in range(images.shape[0]):
                    rows.append({
                        'class_id': int(class_id),
                        'class_name': str(class_name),
                        'img_path': str(img_path[index]),
                        'label': int(label[index].item()),
                        'recon_score': float(recon_scores[index].item()),
                        'memory_score': float(memory_scores[index].item()),
                        'final_score': float(fused_scores[index].item()),
                        'mem_applied': int(memory_result['mem_applied'][index].item()),
                        'routed_sample_idx': int(memory_result['routed_sample_idx'][index].item()),
                        'routed_sample_key': int(memory_result['routed_sample_key'][index].item()),
                        'route_similarity': float(memory_result['route_similarity'][index].item()),
                        'route_angle': float(memory_result['route_angle'][index].item()),
                        'adaptive_angle': float(memory_result['adaptive_angle'][index].item()),
                    })
                    class_row_count += 1

            from sklearn.metrics import average_precision_score, roc_auc_score
            from utils import compute_pro, f1_score_max

            for mode in metric_modes:
                gt_sp_np = torch.cat(class_predictions[mode]['gt_sp']).flatten().numpy()
                pred_sp_np = torch.cat(class_predictions[mode]['pred_sp']).flatten().numpy()
                gt_px_np = torch.cat(class_predictions[mode]['gt_px']).numpy()
                pred_px_np = torch.cat(class_predictions[mode]['pred_px']).numpy()
                per_class_metrics_by_mode[mode].append({
                    'class_id': int(class_id),
                    'class_name': str(class_name),
                    'I-AUROC': float(roc_auc_score(gt_sp_np, pred_sp_np)),
                    'I-AP': float(average_precision_score(gt_sp_np, pred_sp_np)),
                    'I-F1': float(f1_score_max(gt_sp_np, pred_sp_np)),
                    'P-AUROC': float(roc_auc_score(gt_px_np.ravel(), pred_px_np.ravel())),
                    'P-AP': float(average_precision_score(gt_px_np.ravel(), pred_px_np.ravel())),
                    'P-F1': float(f1_score_max(gt_px_np.ravel(), pred_px_np.ravel())),
                    'P-AUPRO': float(compute_pro(gt_px_np, pred_px_np)),
                    'num_samples': int(class_row_count),
                })

        metrics_by_mode = {}
        for mode in metric_modes:
            metrics_df = pd.DataFrame(per_class_metrics_by_mode[mode])
            metrics_by_mode[mode] = {
                'summary': {
                    'I-AUROC': float(metrics_df['I-AUROC'].mean()),
                    'I-AP': float(metrics_df['I-AP'].mean()),
                    'I-F1': float(metrics_df['I-F1'].mean()),
                    'P-AUROC': float(metrics_df['P-AUROC'].mean()),
                    'P-AP': float(metrics_df['P-AP'].mean()),
                    'P-F1': float(metrics_df['P-F1'].mean()),
                    'P-AUPRO': float(metrics_df['P-AUPRO'].mean()),
                },
                'per_class_metrics': per_class_metrics_by_mode[mode],
            }

        score_df = pd.DataFrame(rows)
        summary = dict(metrics_by_mode['fused']['summary'])
        summary.update({
            'num_memory_supports': int(len(memory_bank)),
            'memory_applied_ratio': float(score_df['mem_applied'].mean()) if len(score_df) > 0 else 0.0,
        })
    finally:
        if was_training:
            model.train()
    return score_df, metrics_by_mode['fused']['per_class_metrics'], summary, metrics_by_mode
