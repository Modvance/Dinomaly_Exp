from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image
from torch.utils.data import Dataset


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: Sequence[str], transform):
        self.image_paths = [str(path) for path in image_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image)
        meta = {
            'sample_idx': int(idx),
            'img_path': image_path,
        }
        return tensor, meta


class MultiViewImagePathDataset(Dataset):
    def __init__(self, image_paths: Sequence[str], view_transforms):
        self.image_paths = [str(path) for path in image_paths]
        self.view_transforms = list(view_transforms)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        views = [transform(image.copy()) for transform in self.view_transforms]
        meta = {
            'sample_idx': int(idx),
            'img_path': image_path,
        }
        return views, meta


def discover_image_paths(image_root: str, extensions: Sequence[str], recursive: bool = True, split: str = 'train') -> List[str]:
    root = Path(image_root)
    if not root.is_dir():
        raise FileNotFoundError('image_root not found: {}'.format(image_root))
    allowed = {extension.lower() for extension in extensions}
    requested_split = str(split).lower()
    iterator = root.rglob('*') if recursive else root.glob('*')
    paths = []
    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        parts = {part.lower() for part in path.parts}
        if requested_split != 'all' and requested_split not in parts:
            continue
        paths.append(str(path.resolve()))
    paths.sort()
    if len(paths) == 0:
        raise FileNotFoundError('no images found under {} for split {}'.format(image_root, requested_split))
    return paths


def collate_multiview(batch):
    num_views = len(batch[0][0])
    view_batches = [[] for _ in range(num_views)]
    sample_idx = []
    img_path = []
    for views, meta in batch:
        for view_index, view in enumerate(views):
            view_batches[view_index].append(view)
        sample_idx.append(int(meta['sample_idx']))
        img_path.append(meta['img_path'])
    import torch
    stacked_views = [torch.stack(view_batch, dim=0) for view_batch in view_batches]
    return stacked_views, {
        'sample_idx': sample_idx,
        'img_path': img_path,
    }
