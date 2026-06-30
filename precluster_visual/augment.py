from typing import List

from PIL import Image, ImageOps
from torchvision import transforms


class LetterboxResize:
    def __init__(self, input_size: int, fill: int = 0):
        self.input_size = int(input_size)
        self.fill = int(fill)

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        max_side = max(width, height)
        pad_left = (max_side - width) // 2
        pad_top = (max_side - height) // 2
        pad_right = max_side - width - pad_left
        pad_bottom = max_side - height - pad_top
        squared = ImageOps.expand(image, border=(pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)
        return squared.resize((self.input_size, self.input_size), resample=Image.BICUBIC)


def _build_resize_transform(config):
    input_size = int(config.encoder.input_size)
    if str(config.preprocess.mode).lower() == 'letterbox':
        return LetterboxResize(input_size=input_size, fill=int(config.preprocess.letterbox_fill))
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
    ])


def _scale_crop_transform(input_size: int, scale_min: float, scale_max: float):
    return transforms.Compose([
        transforms.Resize(int(round(input_size * 1.08))),
        transforms.RandomResizedCrop(input_size, scale=(scale_min, scale_max), ratio=(0.95, 1.05)),
    ])


def _mild_color_transform(config):
    input_size = int(config.encoder.input_size)
    base = _build_resize_transform(config)
    return transforms.Compose([
        base,
        transforms.ColorJitter(
            brightness=float(config.multiview.brightness),
            contrast=float(config.multiview.contrast),
            saturation=float(config.multiview.saturation),
            hue=float(config.multiview.hue),
        ),
    ])


def build_view_transforms(config) -> List:
    input_size = int(config.encoder.input_size)
    transforms_list = [_build_resize_transform(config)]
    if config.multiview.enabled:
        if config.multiview.num_views >= 2:
            transforms_list.append(_scale_crop_transform(input_size, float(config.multiview.scale_min), float(config.multiview.scale_max)))
        if config.multiview.num_views >= 3:
            transforms_list.append(_mild_color_transform(config))
    return transforms_list[:max(1, int(config.multiview.num_views if config.multiview.enabled else 1))]
