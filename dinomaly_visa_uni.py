import logging
import os
import random
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.utils.data import ConcatDataset, Subset
from torchvision.datasets import ImageFolder

from dataset import MVTecDataset, TrainDiagDataset, get_data_transforms
from models import vit_encoder
from models.uad import ViTill
from models.vision_transformer import Block as VitBlock, LinearAttention2, bMlp
from optimizers import StableAdamW
from utils import WarmCosineScheduler, evaluation_batch, global_cosine_hm_percent
from warmup_diag import build_prune_plan, load_injected_manifest, parse_warmup_milestones, run_one_warmup_diagnosis, save_prune_plan

import argparse
import warnings

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


def build_train_dataloader(train_datasets, batch_size, num_workers):
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


def train(item_list):
    setup_seed(1)

    total_iters = 10000
    batch_size = 16
    num_workers = 4
    image_size = 448
    crop_size = 392
    train_start_time = time.time()
    final_eval_summary = None

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_data_list = []
    test_data_list = []
    contaminated_paths, manifest_path = (None, None)
    if args.warmup_diag or args.warmup_denoise:
        contaminated_paths, manifest_path = load_injected_manifest(
            args.data_path,
            os.path.dirname(__file__),
            manifest_path=args.diag_manifest_path,
        )
        if contaminated_paths is None:
            print_fn('warmup diagnosis: inject_defects manifest not found, contamination-aware metrics will be skipped.')
        else:
            print_fn('warmup diagnosis: loaded contamination manifest {}.'.format(manifest_path))

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

    train_data, train_dataloader = build_train_dataloader(train_data_list, batch_size=batch_size, num_workers=num_workers)

    train_eval_dataloader = None
    if args.warmup_diag or args.warmup_denoise:
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
            attn_drop=0.,
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

    has_pruned = False
    it = 0
    for _ in range(int(np.ceil(total_iters / len(train_dataloader)))):
        model.train()
        loss_list = []

        for img, label in train_dataloader:
            img = img.to(device)
            label = label.to(device)

            en, de = model(img)

            p_final = 0.9
            p = min(p_final * it / 1000, p_final)
            loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)

            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()

            current_iter = it + 1
            if args.warmup_diag and current_iter in args.warmup_milestones and not (args.warmup_denoise and current_iter == args.warmup_end_iter):
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

            if args.warmup_denoise and (not has_pruned) and current_iter == args.warmup_end_iter:
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
                    save_scores=args.warmup_save_scores_before_prune,
                )
                scored_df = diagnosis_result['df'].copy()

                class_offsets = {}
                offset = 0
                for class_id, dataset in enumerate(train_data_list):
                    class_offsets[class_id] = offset
                    offset += len(dataset)
                scored_df['local_idx'] = scored_df.apply(
                    lambda row: int(row['sample_idx']) - class_offsets[int(row['class_id'])],
                    axis=1,
                )

                prune_plan = build_prune_plan(
                    scored_df,
                    prune_ratio=args.warmup_prune_ratio,
                    min_keep_per_class=args.warmup_min_keep_per_class,
                )
                save_prune_plan(diagnosis_result['iter_dir'], prune_plan, current_iter)

                pruned_train_data_list = []
                for class_id, train_dataset in enumerate(train_data_list):
                    retained_indices = prune_plan['retained_index_map'].get(class_id)
                    if retained_indices is None:
                        retained_indices = list(range(len(train_dataset)))
                    pruned_train_data_list.append(Subset(train_dataset, retained_indices))

                train_data_list = pruned_train_data_list
                train_data, train_dataloader = build_train_dataloader(
                    train_data_list,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
                train_eval_dataloader = build_train_eval_dataloader(
                    train_data_list,
                    data_root=args.data_path,
                    contaminated_paths=contaminated_paths,
                    batch_size=args.diag_batch_size,
                    num_workers=args.diag_num_workers,
                    item_list=item_list,
                )
                has_pruned = True
                print_fn('warmup denoise iter {}: pruned {} / {} training samples.'.format(
                    current_iter,
                    prune_plan['summary']['num_pruned'],
                    prune_plan['summary']['num_samples_before_prune'],
                ))
                for class_name, class_stats in prune_plan['summary']['class_counts'].items():
                    print_fn('  {}: removed {}, retained {}'.format(
                        class_name,
                        class_stats['removed'],
                        class_stats['retained'],
                    ))
                print_fn('train image number after prune:{}'.format(len(train_data)))
                model.train()

            if current_iter % 5000 == 0:
                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                for item, test_data in zip(item_list, test_data_list):
                    test_dataloader = torch.utils.data.DataLoader(
                        test_data,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
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
            if (it + 1) % 100 == 0:
                print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
                loss_list = []

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
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='../VisA_pytorch/1cls')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument(
        '--save_name',
        type=str,
        default='vitill_visa_uni_dinov2br_c392r_en29_bn4dp2_de8_laelu_md2_i1_it10k_sams2e3_wd1e4_w1hcosa_ghmp09f01w01_b16_ev_s1',
    )
    parser.add_argument('--warmup_diag', action='store_true')
    parser.add_argument('--warmup_denoise', action='store_true')
    parser.add_argument('--warmup_milestones', type=str, default='50, 100, 200, 400, 600,1000,2000,4000')
    parser.add_argument('--warmup_end_iter', type=int, default=1000)
    parser.add_argument('--warmup_prune_ratio', type=float, default=0.05)
    parser.add_argument('--warmup_min_keep_per_class', type=int, default=20)
    parser.add_argument('--warmup_save_scores_before_prune', action='store_true')
    parser.add_argument('--diag_save_dir', type=str, default='./warmup_diag')
    parser.add_argument('--diag_manifest_path', type=str, default=None)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--diag_batch_size', type=int, default=16)
    parser.add_argument('--diag_num_workers', type=int, default=4)
    parser.add_argument('--diag_max_ratio', type=float, default=0.01)
    parser.add_argument('--diag_resize_mask', type=int, default=256)
    args = parser.parse_args()
    args.warmup_milestones = parse_warmup_milestones(args.warmup_milestones)

    item_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    if not os.path.isabs(args.diag_save_dir):
        args.diag_save_dir = os.path.join(args.save_dir, args.save_name, args.diag_save_dir)

    device = 'cuda:{}'.format(args.gpus) if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    train(item_list)
