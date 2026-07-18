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
    if encoder is None or not hasattr(encoder, 'prepare_tokens_with_masks'):
        raise ValueError('TailGuard-B pseudo-class memory requires a DINOv2-style encoder')
    tokens = encoder.prepare_tokens_with_masks(images)
    for block in encoder.blocks:
        tokens = block(tokens)
    return _normalize_rows(encoder.norm(tokens)[:, 0])


def _allocate_patch_quotas(capacities: List[int], max_patches: int) -> List[int]:
    total_patches = int(sum(capacities))
    target = total_patches if max_patches <= 0 else min(total_patches, int(max_patches))
    quotas = [0] * len(capacities)
    active = [index for index, capacity in enumerate(capacities) if capacity > 0]
    remaining = target
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        next_active = []
        for index in active:
            added = min(share, capacities[index] - quotas[index], remaining)
            quotas[index] += added
            remaining -= added
            if quotas[index] < capacities[index]:
                next_active.append(index)
            if remaining == 0:
                next_active.extend(active[active.index(index) + 1:])
                break
        active = next_active
    return quotas


def _sample_patch_features(features: torch.Tensor, quota: int) -> torch.Tensor:
    if quota <= 0:
        return features[:0]
    if quota >= features.shape[0]:
        return features
    indices = torch.linspace(0, features.shape[0] - 1, steps=quota).round().long()
    return features.index_select(0, indices)


def _registry_state(members_df: pd.DataFrame, classes_df: pd.DataFrame):
    required_members = {'sample_idx', 'img_path', 'pseudo_class_id'}
    required_classes = {'pseudo_class_id', 'pseudo_class_type', 'num_members'}
    missing_members = required_members.difference(members_df.columns)
    missing_classes = required_classes.difference(classes_df.columns)
    if missing_members:
        raise ValueError('pseudo-class members are missing: {}'.format(', '.join(sorted(missing_members))))
    if missing_classes:
        raise ValueError('pseudo-classes are missing: {}'.format(', '.join(sorted(missing_classes))))
    members = members_df.copy().reset_index(drop=True)
    classes = classes_df.copy().reset_index(drop=True)
    members['sample_idx'] = members['sample_idx'].astype(int)
    members['pseudo_class_id'] = members['pseudo_class_id'].astype(int)
    classes['pseudo_class_id'] = classes['pseudo_class_id'].astype(int)
    if members['sample_idx'].duplicated().any() or members['img_path'].astype(str).duplicated().any():
        raise ValueError('pseudo-class members must have unique sample_idx and img_path')
    if classes['pseudo_class_id'].duplicated().any() or (classes['num_members'].astype(int) <= 0).any():
        raise ValueError('pseudo-class registry has invalid classes')
    counts = members.groupby('pseudo_class_id').size().to_dict()
    expected_counts = classes.set_index('pseudo_class_id')['num_members'].astype(int).to_dict()
    if counts != expected_counts:
        raise ValueError('pseudo-class membership counts do not match class registry')
    return members, classes


@torch.no_grad()
def build_tailguard_b_pseudoclass_memory_system(model,
                                                train_eval_dataloader,
                                                device,
                                                members_df: pd.DataFrame,
                                                classes_df: pd.DataFrame,
                                                args) -> Dict:
    members, classes = _registry_state(members_df, classes_df)
    class_type_lookup = classes.set_index('pseudo_class_id')['pseudo_class_type'].to_dict()
    member_lookup = {
        int(row.sample_idx): {
            'pseudo_class_id': int(row.pseudo_class_id),
            'img_path': str(row.img_path),
        }
        for row in members.itertuples(index=False)
    }
    observed = {}
    was_training = model.training
    model.eval()
    try:
        for images, _, meta in train_eval_dataloader:
            images = images.to(device)
            patch_features, spatial_size = extract_encoder_patch_tokens(model, images)
            cls_embeddings = _extract_encoder_cls_embeddings(model, images)
            sample_indices = meta['sample_idx']
            for index in range(images.shape[0]):
                sample_idx = int(sample_indices[index])
                member = member_lookup.get(sample_idx)
                if member is None:
                    continue
                if str(meta['img_path'][index]) != member['img_path']:
                    raise ValueError('pseudo-class member image path does not match train-eval metadata')
                if sample_idx in observed:
                    raise ValueError('train-eval loader yielded a pseudo-class member more than once')
                record = {
                    'pseudo_class_id': member['pseudo_class_id'],
                    'img_path': member['img_path'],
                    'cls': cls_embeddings[index].detach().cpu().contiguous(),
                }
                if class_type_lookup[member['pseudo_class_id']] == 'tail':
                    record.update({
                        'patches': patch_features[index].detach().cpu().contiguous(),
                        'spatial_size': tuple(int(value) for value in spatial_size),
                    })
                observed[sample_idx] = record
    finally:
        if was_training:
            model.train()

    missing_members = sorted(set(member_lookup).difference(observed))
    if missing_members:
        raise ValueError('train-eval loader is missing pseudo-class members: {}'.format(missing_members[:10]))

    max_patches = int(getattr(args, 'tgb_mem_max_patches_per_class', 20000))
    class_entries = {}
    for class_row in classes.sort_values('pseudo_class_id', kind='mergesort').itertuples(index=False):
        pseudo_class_id = int(class_row.pseudo_class_id)
        pseudo_class_type = str(class_row.pseudo_class_type)
        class_members = members.loc[members['pseudo_class_id'] == pseudo_class_id].sort_values(
            ['img_path', 'sample_idx'], kind='mergesort'
        )
        records = [observed[int(sample_idx)] for sample_idx in class_members['sample_idx'].astype(int)]
        cls_matrix = torch.stack([record['cls'] for record in records], dim=0)
        prototype = _normalize_rows(cls_matrix.mean(dim=0, keepdim=True))[0].cpu().contiguous()
        entry = {
            'pseudo_class_id': pseudo_class_id,
            'pseudo_class_type': pseudo_class_type,
            'num_members': int(len(records)),
            'member_sample_idx': class_members['sample_idx'].astype(int).tolist(),
            'prototype_cls': prototype,
            'has_memory_bank': pseudo_class_type == 'tail',
            'num_patches': 0,
            'feature_dim': 0,
            'spatial_size': None,
            'member_patch_quotas': [],
        }
        if pseudo_class_type == 'tail':
            feature_dimensions = {int(record['patches'].shape[1]) for record in records}
            spatial_sizes = {record['spatial_size'] for record in records}
            if len(feature_dimensions) != 1 or len(spatial_sizes) != 1:
                raise ValueError('tail pseudo-class members have inconsistent patch shapes')
            capacities = [int(record['patches'].shape[0]) for record in records]
            quotas = _allocate_patch_quotas(capacities, max_patches)
            selected = [
                _sample_patch_features(record['patches'], quota)
                for record, quota in zip(records, quotas)
                if quota > 0
            ]
            if len(selected) == 0:
                raise ValueError('tail pseudo-class memory bank has no patch tokens')
            features = torch.cat(selected, dim=0).contiguous().cpu()
            entry.update({
                'features': features,
                'num_patches': int(features.shape[0]),
                'feature_dim': int(features.shape[1]),
                'spatial_size': next(iter(spatial_sizes)),
                'member_patch_quotas': [
                    {
                        'sample_idx': int(sample_idx),
                        'num_available_patches': int(capacity),
                        'num_selected_patches': int(quota),
                    }
                    for sample_idx, capacity, quota in zip(
                        class_members['sample_idx'].astype(int), capacities, quotas
                    )
                ],
            })
        elif pseudo_class_type != 'head':
            raise ValueError('unknown pseudo-class type: {}'.format(pseudo_class_type))
        class_entries[pseudo_class_id] = entry

    return {
        'schema_version': 1,
        'class_entries': class_entries,
        'num_pseudo_classes': int(len(class_entries)),
        'num_head_pseudo_classes': int(sum(entry['pseudo_class_type'] == 'head' for entry in class_entries.values())),
        'num_tail_pseudo_classes': int(sum(entry['pseudo_class_type'] == 'tail' for entry in class_entries.values())),
        'num_tail_memory_banks': int(sum(entry['has_memory_bank'] for entry in class_entries.values())),
        'max_patches_per_class': max_patches,
    }


def _routing_cache(memory_system: Dict, device):
    entries = memory_system.get('class_entries', {})
    if len(entries) == 0:
        raise ValueError('pseudo-class memory system has no classes')
    class_ids = sorted(int(class_id) for class_id in entries)
    prototypes = torch.stack([entries[class_id]['prototype_cls'] for class_id in class_ids], dim=0).to(device)
    return {
        'class_ids': class_ids,
        'prototypes': _normalize_rows(prototypes),
        'class_types': [str(entries[class_id]['pseudo_class_type']) for class_id in class_ids],
    }


@torch.no_grad()
def compute_pseudoclass_memory_scores_for_batch(model,
                                                images: torch.Tensor,
                                                memory_system: Dict,
                                                args,
                                                device,
                                                routing_cache=None,
                                                device_memory_cache=None):
    images = images.to(device)
    patch_features, spatial_size = extract_encoder_patch_tokens(model, images)
    cls_embeddings = _extract_encoder_cls_embeddings(model, images)
    cache = _routing_cache(memory_system, device) if routing_cache is None else routing_cache
    similarities = cls_embeddings @ cache['prototypes'].T
    best_positions = similarities.argmax(dim=1)
    best_similarity = similarities.gather(1, best_positions[:, None])[:, 0]
    if similarities.shape[1] > 1:
        second_similarity = torch.topk(similarities, k=2, dim=1).values[:, 1]
    else:
        second_similarity = torch.full_like(best_similarity, float('nan'))
    chunk_size = int(getattr(args, 'tgb_mem_chunk_size', getattr(args, 'mem_chunk_size', 4096)))
    topk_ratio = float(getattr(args, 'tgb_memory_topk_ratio', getattr(args, 'mem_topk_ratio', 0.05)))
    entries = memory_system['class_entries']
    scores = []
    maps = []
    mem_applied = []
    predicted_ids = []
    predicted_types = []
    for index, best_position in enumerate(best_positions.tolist()):
        pseudo_class_id = int(cache['class_ids'][best_position])
        pseudo_class_type = cache['class_types'][best_position]
        predicted_ids.append(pseudo_class_id)
        predicted_types.append(pseudo_class_type)
        if pseudo_class_type == 'head':
            scores.append(torch.tensor(0.0, device=device))
            maps.append(torch.zeros(spatial_size, device=device))
            mem_applied.append(0)
            continue
        entry = entries[pseudo_class_id]
        if not entry.get('has_memory_bank') or 'features' not in entry:
            raise ValueError('predicted tail pseudo-class has no memory bank: {}'.format(pseudo_class_id))
        memory_features = entry['features']
        if device_memory_cache is not None:
            memory_features = device_memory_cache.setdefault(
                pseudo_class_id,
                entry['features'].to(device),
            )
        patch_scores = compute_memory_patch_scores(
            patch_features[index],
            memory_features,
            chunk_size=chunk_size,
        )
        scores.append(pool_patch_scores(patch_scores, topk_ratio=topk_ratio))
        maps.append(patch_scores.reshape(spatial_size))
        mem_applied.append(1)
    return {
        'memory_scores': torch.stack(scores),
        'memory_patch_maps': torch.stack(maps),
        'mem_applied': torch.tensor(mem_applied, dtype=torch.int64),
        'predicted_pseudo_class_id': torch.tensor(predicted_ids, dtype=torch.int64),
        'predicted_pseudo_class_type': predicted_types,
        'top1_similarity': best_similarity.detach().cpu(),
        'second_similarity': second_similarity.detach().cpu(),
        'similarity_margin': (best_similarity - second_similarity).detach().cpu(),
    }


@torch.no_grad()
def run_pseudoclass_memory_evaluation(model,
                                      test_data_list,
                                      item_list,
                                      memory_system: Dict,
                                      args,
                                      device):
    was_training = model.training
    model.eval()
    try:
        gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
        rows: List[dict] = []
        lambda_value = float(getattr(args, 'tgb_memory_fusion_lambda', 1.0))
        batch_size = int(args.batch_size)
        num_workers = int(args.num_workers)
        max_ratio = float(getattr(args, 'diag_max_ratio', getattr(args, 'max_ratio', 0.01)))
        resize_mask = int(getattr(args, 'diag_resize_mask', getattr(args, 'resize_mask', 256)))
        metric_modes = ('recon', 'memory', 'fused')
        per_class_metrics_by_mode = {mode: [] for mode in metric_modes}
        routing_cache = _routing_cache(memory_system, device)
        device_memory_cache = {}

        for class_id, (class_name, test_data) in enumerate(zip(item_list, test_data_list)):
            loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            predictions = {
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
                memory_result = compute_pseudoclass_memory_scores_for_batch(
                    model,
                    images,
                    memory_system,
                    args,
                    device,
                    routing_cache=routing_cache,
                    device_memory_cache=device_memory_cache,
                )
                memory_scores = memory_result['memory_scores']
                memory_maps = resize_memory_map(
                    memory_result['memory_patch_maps'].to(device), recon_map.shape[-2:]
                ).unsqueeze(1)
                tail_mask = memory_result['mem_applied'].to(device=device, dtype=torch.bool)
                fused_scores = torch.where(tail_mask, recon_scores + lambda_value * memory_scores, recon_scores)
                fused_maps = torch.where(
                    tail_mask[:, None, None, None],
                    recon_map + lambda_value * memory_maps,
                    recon_map,
                )
                memory_mode_scores = torch.where(tail_mask, memory_scores, recon_scores)
                memory_mode_maps = torch.where(
                    tail_mask[:, None, None, None], memory_maps, recon_map
                )
                scores_by_mode = {'recon': recon_scores, 'memory': memory_mode_scores, 'fused': fused_scores}
                maps_by_mode = {'recon': recon_map, 'memory': memory_mode_maps, 'fused': fused_maps}

                gt_bool = gt.bool()
                if gt_bool.shape[1] > 1:
                    gt_bool = torch.max(gt_bool, dim=1, keepdim=True)[0]
                for mode in metric_modes:
                    predictions[mode]['gt_sp'].append(label.cpu())
                    predictions[mode]['pred_sp'].append(scores_by_mode[mode].detach().cpu())
                    predictions[mode]['gt_px'].append(gt_bool[:, 0].detach().cpu())
                    predictions[mode]['pred_px'].append(maps_by_mode[mode][:, 0].detach().cpu())
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
                        'predicted_pseudo_class_id': int(memory_result['predicted_pseudo_class_id'][index].item()),
                        'predicted_pseudo_class_type': memory_result['predicted_pseudo_class_type'][index],
                        'top1_similarity': float(memory_result['top1_similarity'][index].item()),
                        'second_similarity': float(memory_result['second_similarity'][index].item()),
                        'similarity_margin': float(memory_result['similarity_margin'][index].item()),
                    })
                    class_row_count += 1

            from sklearn.metrics import average_precision_score, roc_auc_score
            from utils import compute_pro, f1_score_max
            for mode in metric_modes:
                gt_sp_np = torch.cat(predictions[mode]['gt_sp']).flatten().numpy()
                pred_sp_np = torch.cat(predictions[mode]['pred_sp']).flatten().numpy()
                gt_px_np = torch.cat(predictions[mode]['gt_px']).numpy()
                pred_px_np = torch.cat(predictions[mode]['pred_px']).numpy()
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
            'num_pseudo_classes': int(memory_system['num_pseudo_classes']),
            'num_head_pseudo_classes': int(memory_system['num_head_pseudo_classes']),
            'num_tail_pseudo_classes': int(memory_system['num_tail_pseudo_classes']),
            'num_tail_memory_banks': int(memory_system['num_tail_memory_banks']),
            'memory_applied_ratio': float(score_df['mem_applied'].mean()) if len(score_df) > 0 else 0.0,
        })
    finally:
        if was_training:
            model.train()
    return score_df, metrics_by_mode['fused']['per_class_metrics'], summary, metrics_by_mode
