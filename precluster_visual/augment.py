from typing import List

from torchvision import transforms


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _base_transform(input_size: int):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _scale_crop_transform(input_size: int, scale_min: float, scale_max: float):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomResizedCrop((input_size, input_size), scale=(scale_min, scale_max), ratio=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _mild_color_transform(input_size: int, brightness: float, contrast: float):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.CenterCrop(input_size),
        transforms.ColorJitter(brightness=brightness, contrast=contrast),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def build_view_transforms(config) -> List[transforms.Compose]:
    input_size = int(config.encoder.input_size)
    transforms_list = [_base_transform(input_size)]
    if config.multiview.enabled:
        if config.multiview.num_views >= 2:
            transforms_list.append(_scale_crop_transform(input_size, config.multiview.scale_min, config.multiview.scale_max))
        if config.multiview.num_views >= 3:
            transforms_list.append(_mild_color_transform(input_size, config.multiview.brightness, config.multiview.contrast))
    return transforms_list[:max(1, int(config.multiview.num_views if config.multiview.enabled else 1))]
