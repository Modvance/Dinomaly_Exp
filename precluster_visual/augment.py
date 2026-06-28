from typing import List

from torchvision import transforms


def _base_transform(input_size: int):
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
    ])


def _scale_crop_transform(input_size: int, scale_min: float, scale_max: float):
    return transforms.Compose([
        transforms.Resize(int(round(input_size * 1.08))),
        transforms.RandomResizedCrop(input_size, scale=(scale_min, scale_max), ratio=(0.95, 1.05)),
    ])


def _mild_color_transform(input_size: int, brightness: float, contrast: float, saturation: float, hue: float):
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        ),
    ])


def build_view_transforms(config) -> List[transforms.Compose]:
    input_size = int(config.encoder.input_size)
    transforms_list = [_base_transform(input_size)]
    if config.multiview.enabled:
        if config.multiview.num_views >= 2:
            transforms_list.append(_scale_crop_transform(input_size, float(config.multiview.scale_min), float(config.multiview.scale_max)))
        if config.multiview.num_views >= 3:
            transforms_list.append(_mild_color_transform(
                input_size,
                float(config.multiview.brightness),
                float(config.multiview.contrast),
                float(config.multiview.saturation),
                float(config.multiview.hue),
            ))
    return transforms_list[:max(1, int(config.multiview.num_views if config.multiview.enabled else 1))]
