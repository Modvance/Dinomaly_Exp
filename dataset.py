import random

from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import torch.multiprocessing
import json
from torch.utils.data import Dataset, Subset

# import imgaug.augmenters as iaa
# from perlin import rand_perlin_2d_np

torch.multiprocessing.set_sharing_strategy('file_system')


def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms


def get_strong_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomResizedCrop((isize, isize), scale=(0.6, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    return data_transforms


class _BaseImageFolderMetaDataset(Dataset):
    def __init__(self, dataset, data_root, class_name, class_id):
        self.dataset = dataset
        self.data_root = os.path.abspath(data_root)
        self.class_name = class_name
        self.class_id = class_id

    def __len__(self):
        return len(self.dataset)

    def _resolve_base_sample(self, idx):
        if isinstance(self.dataset, Subset):
            base_idx = self.dataset.indices[idx]
            base_dataset = self.dataset.dataset
        else:
            base_idx = idx
            base_dataset = self.dataset
        return base_dataset, base_idx

    def _build_base_meta(self, idx):
        base_dataset, base_idx = self._resolve_base_sample(idx)
        img_path = base_dataset.samples[base_idx][0]
        rel_path = os.path.relpath(img_path, self.data_root).replace('\\', '/')
        return {
            'img_path': img_path,
            'rel_path': rel_path,
            'class_name': self.class_name,
            'class_id': self.class_id,
            'base_idx': int(base_idx),
        }


class TrainDiagDataset(_BaseImageFolderMetaDataset):
    def __init__(self, dataset, data_root, class_name, class_id, sample_offset=0, contaminated_paths=None):
        super().__init__(dataset, data_root, class_name, class_id)
        self.sample_offset = sample_offset
        self.contaminated_paths = contaminated_paths

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        meta = self._build_base_meta(idx)

        is_contaminated = None
        if self.contaminated_paths is not None:
            is_contaminated = int(meta['rel_path'] in self.contaminated_paths)

        meta.update({
            'sample_idx': self.sample_offset + idx,
            'is_contaminated': -1 if is_contaminated is None else is_contaminated,
        })
        meta.pop('rel_path', None)
        return image, label, meta


class TrainWeightDataset(_BaseImageFolderMetaDataset):
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        meta = self._build_base_meta(idx)
        meta.pop('rel_path', None)
        return image, label, meta


class TrainPatchWeightDataset(_BaseImageFolderMetaDataset):
    def __init__(self, dataset, data_root, class_name, class_id, sample_offset=0, patch_weight_bank=None,
                 default_patch_grid_size=(32, 32), skip_patch_denoise=False):
        super().__init__(dataset, data_root, class_name, class_id)
        self.sample_offset = int(sample_offset)
        self.patch_weight_bank = {} if patch_weight_bank is None else patch_weight_bank
        self.default_patch_grid_size = tuple(int(v) for v in default_patch_grid_size)
        self.skip_patch_denoise = bool(skip_patch_denoise)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        meta = self._build_base_meta(idx)
        sample_idx = self.sample_offset + idx
        bank_entry = None if self.skip_patch_denoise else self.patch_weight_bank.get(int(sample_idx))
        if bank_entry is None:
            patch_weight = torch.ones(self.default_patch_grid_size, dtype=torch.float32)
            patch_active = False
        else:
            patch_weight = bank_entry['weight'].detach().cpu().float()
            patch_active = bool(bank_entry.get('active', False))
        meta.update({
            'sample_idx': int(sample_idx),
            'patch_weight': patch_weight,
            'patch_active': int(patch_active),
        })
        meta.pop('rel_path', None)
        return image, label, meta


class TrainPhase5WeightDataset(_BaseImageFolderMetaDataset):
    def __init__(self, dataset, data_root, class_name, class_id, sample_offset=0, reliability_bank=None,
                 default_patch_grid_size=(32, 32), skip_phase5_weighting=False):
        super().__init__(dataset, data_root, class_name, class_id)
        self.sample_offset = int(sample_offset)
        self.reliability_bank = {} if reliability_bank is None else reliability_bank
        if isinstance(default_patch_grid_size, str):
            parts = [part.strip() for part in default_patch_grid_size.split(',') if part.strip()]
            if len(parts) == 1:
                self.default_patch_grid_size = (int(parts[0]), int(parts[0]))
            else:
                self.default_patch_grid_size = (int(parts[0]), int(parts[1]))
        else:
            self.default_patch_grid_size = tuple(int(v) for v in default_patch_grid_size)
        self.skip_phase5_weighting = bool(skip_phase5_weighting)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        meta = self._build_base_meta(idx)
        sample_idx = self.sample_offset + idx
        bank_entry = None if self.skip_phase5_weighting else self.reliability_bank.get(int(sample_idx))
        if bank_entry is None:
            w_img = torch.tensor(1.0, dtype=torch.float32)
            w_patch = torch.ones(self.default_patch_grid_size, dtype=torch.float32)
            group_id = -1
        else:
            w_img = torch.tensor(float(bank_entry.get('w_img', 1.0)), dtype=torch.float32)
            w_patch = bank_entry['w_patch'].detach().cpu().float()
            group_id = int(bank_entry.get('group_id', -1))
        meta.update({
            'sample_idx': int(sample_idx),
            'group_id': int(group_id),
            'w_img': w_img,
            'w_patch': w_patch,
        })
        meta.pop('rel_path', None)
        return image, label, meta


PHASE6_AUG_TYPES = (
    'hflip',
    'rotate_small',
    'brightness_contrast',
    'translate_small',
    'resized_crop_weak',
)


def _phase6_clean_value(value, default=None):
    if value is None:
        return default
    if isinstance(value, float) and np.isnan(value):
        return default
    return value


def _phase6_apply_weak_transform(image, aug_type, aug_seed):
    aug_type = str(aug_type)
    rng = random.Random(int(aug_seed))
    if aug_type == 'hflip':
        return TF.hflip(image)
    if aug_type == 'rotate_small':
        angle = rng.uniform(-8.0, 8.0)
        return TF.rotate(image, angle=angle, interpolation=transforms.InterpolationMode.BILINEAR)
    if aug_type == 'brightness_contrast':
        brightness = rng.uniform(0.92, 1.08)
        contrast = rng.uniform(0.92, 1.08)
        image = TF.adjust_brightness(image, brightness)
        return TF.adjust_contrast(image, contrast)
    if aug_type == 'translate_small':
        max_dx = max(1, int(round(image.size[0] * 0.04)))
        max_dy = max(1, int(round(image.size[1] * 0.04)))
        translate = (rng.randint(-max_dx, max_dx), rng.randint(-max_dy, max_dy))
        return TF.affine(
            image,
            angle=0.0,
            translate=translate,
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
    if aug_type == 'resized_crop_weak':
        width, height = image.size
        crop_scale = rng.uniform(0.92, 1.0)
        crop_h = max(1, int(round(height * crop_scale)))
        crop_w = max(1, int(round(width * crop_scale)))
        max_top = max(0, height - crop_h)
        max_left = max(0, width - crop_w)
        top = 0 if max_top == 0 else rng.randint(0, max_top)
        left = 0 if max_left == 0 else rng.randint(0, max_left)
        return TF.resized_crop(
            image,
            top=top,
            left=left,
            height=crop_h,
            width=crop_w,
            size=(height, width),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
    return image


class TrainPhase6Dataset(_BaseImageFolderMetaDataset):
    def __init__(self, dataset, data_root, class_name, class_id, sample_rows=None):
        super().__init__(dataset, data_root, class_name, class_id)
        self.sample_rows = [] if sample_rows is None else [dict(row) for row in sample_rows]
        self.base_idx_to_local_idx = {}
        for local_idx in range(len(self.dataset)):
            base_meta = self._build_base_meta(local_idx)
            self.base_idx_to_local_idx[int(base_meta['base_idx'])] = int(local_idx)

    def __len__(self):
        return len(self.sample_rows)

    def _load_parent_image(self, base_idx, aug_type=None, aug_seed=None):
        local_idx = self.base_idx_to_local_idx[int(base_idx)]
        base_dataset, resolved_base_idx = self._resolve_base_sample(local_idx)
        img_path = base_dataset.samples[resolved_base_idx][0]
        image = Image.open(img_path).convert('RGB')
        if aug_type:
            image = _phase6_apply_weak_transform(image, aug_type=aug_type, aug_seed=aug_seed)
        transform = getattr(base_dataset, 'transform', None)
        if transform is not None:
            image = transform(image)
        else:
            image = TF.to_tensor(image)
        return image, int(self.class_id), local_idx

    def __getitem__(self, idx):
        row = self.sample_rows[idx]
        is_augmented = int(_phase6_clean_value(row.get('is_augmented'), 0))
        base_idx = int(_phase6_clean_value(row.get('base_idx'), -1))
        if base_idx < 0:
            raise ValueError('phase6 sample row is missing base_idx')

        if is_augmented:
            aug_type = str(_phase6_clean_value(row.get('aug_type'), 'identity'))
            aug_seed = int(_phase6_clean_value(row.get('aug_seed'), 0))
            image, label, local_idx = self._load_parent_image(base_idx, aug_type=aug_type, aug_seed=aug_seed)
        else:
            local_idx = self.base_idx_to_local_idx[int(base_idx)]
            image, label = self.dataset[local_idx]
            aug_type = str(_phase6_clean_value(row.get('aug_type'), ''))
            aug_seed = int(_phase6_clean_value(row.get('aug_seed'), -1))

        meta = self._build_base_meta(local_idx)
        meta.update({
            'sample_idx': int(_phase6_clean_value(row.get('sample_idx'), idx)),
            'sample_key': str(_phase6_clean_value(row.get('sample_key'), 'sample::{}'.format(idx))),
            'group_id': int(_phase6_clean_value(row.get('group_id'), -1)),
            'pseudo_group_id': int(_phase6_clean_value(row.get('pseudo_group_id'), _phase6_clean_value(row.get('group_id'), -1))),
            'sampler_weight': float(_phase6_clean_value(row.get('sampler_weight'), 1.0)),
            'loss_weight': float(_phase6_clean_value(row.get('loss_weight'), 1.0)),
            'is_augmented': int(is_augmented),
            'aug_parent_key': str(_phase6_clean_value(row.get('aug_parent_key'), '')),
            'aug_type': aug_type,
            'aug_seed': int(aug_seed),
            'aug_rank': int(_phase6_clean_value(row.get('aug_rank'), -1)),
            'selected_group_col': str(_phase6_clean_value(row.get('selected_group_col'), 'group_id')),
        })
        meta.pop('rel_path', None)
        return image, label, meta


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class RealIADDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, transform, gt_transform, phase):
        self.img_path = os.path.join(root, 'realiad_1024', category)
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase

        json_path = os.path.join(root, 'realiad_jsons', 'realiad_jsons', category + '.json')
        with open(json_path) as file:
            class_json = file.read()
        class_json = json.loads(class_json)

        self.img_paths, self.gt_paths, self.labels, self.types = [], [], [], []

        data_set = class_json[phase]
        for sample in data_set:
            self.img_paths.append(os.path.join(root, 'realiad_1024', category, sample['image_path']))
            label = sample['anomaly_class'] != 'OK'
            if label:
                self.gt_paths.append(os.path.join(root, 'realiad_1024', category, sample['mask_path']))
            else:
                self.gt_paths.append(None)
            self.labels.append(label)
            self.types.append(sample['anomaly_class'])

        self.img_paths = np.array(self.img_paths)
        self.gt_paths = np.array(self.gt_paths)
        self.labels = np.array(self.labels)
        self.types = np.array(self.types)
        self.cls_idx = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if self.phase == 'train':
            return img, label

        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class LOCODataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*/000.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        size = (img.size[1], img.size[0])
        img = self.transform(img)
        type = self.types[idx]
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path, type, size


class InsPLADDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
        self.transform = transform
        self.phase = phase
        # load dataset
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        tot_labels = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
                img_tot_paths.extend(img_paths)
                tot_labels.extend([0] * len(img_paths))
            else:
                if self.phase == 'train':
                    continue
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
                img_tot_paths.extend(img_paths)
                tot_labels.extend([1] * len(img_paths))

        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        img = Image.open(img_path).convert('RGB')

        img = self.transform(img)

        return img, label, img_path


class AeBADDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.phase = phase
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        defect_types = [i for i in defect_types if i[0] != '.']
        for defect_type in defect_types:
            if defect_type == 'good':
                domain_types = os.listdir(os.path.join(self.img_path, defect_type))
                domain_types = [i for i in domain_types if i[0] != '.']

                for domain_type in domain_types:
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type, domain_type) + "/*.png")
                    img_tot_paths.extend(img_paths)
                    gt_tot_paths.extend([0] * len(img_paths))
                    tot_labels.extend([0] * len(img_paths))
                    tot_types.extend(['good'] * len(img_paths))
            else:
                domain_types = os.listdir(os.path.join(self.img_path, defect_type))
                domain_types = [i for i in domain_types if i[0] != '.']

                for domain_type in domain_types:
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type, domain_type) + "/*.png")
                    gt_paths = glob.glob(os.path.join(self.gt_path, defect_type, domain_type) + "/*.png")
                    img_paths.sort()
                    gt_paths.sort()
                    img_tot_paths.extend(img_paths)
                    gt_tot_paths.extend(gt_paths)
                    tot_labels.extend([1] * len(img_paths))
                    tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if self.phase == 'train':
            return img, label
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class MiniDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):

        self.img_path = root
        self.transform = transform
        # load dataset
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        tot_labels = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
            img_tot_paths.extend(img_paths)
            tot_labels.extend([1] * len(img_paths))

        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
            img_path, label = self.img_paths[idx], self.labels[idx]
            img = Image.open(img_path).convert('RGB')
        except:
            img_path, label = self.img_paths[idx - 1], self.labels[idx - 1]
            img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, label


class MVTecDRAEMDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, strong_transform, phase, anomaly_source_path, anomaly_ratio=0.5,
                 size=256):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        self.strong_transform = strong_transform
        self.anomaly_ratio = anomaly_ratio
        self.size = size
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path + "/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path):
        no_anomaly = random.random()
        if no_anomaly > self.anomaly_ratio:
            return image, 0
        else:
            aug = self.randAugmenter()

            perlin_scale = 6
            min_perlin_scale = 0
            anomaly_source_img = Image.open(anomaly_source_path).convert('RGB').resize((self.size, self.size))
            anomaly_source_img = np.asarray(anomaly_source_img)
            anomaly_img_augmented = aug(image=anomaly_source_img)

            perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
            perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

            perlin_noise = rand_perlin_2d_np((self.size, self.size),
                                             (perlin_scalex, perlin_scaley))
            perlin_noise = self.rot(image=perlin_noise)
            threshold = 0.5
            perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
            perlin_thr = np.expand_dims(perlin_thr, axis=2)

            img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr

            beta = random.random() * 0.7 + 0.1

            image = image.resize((self.size, self.size))
            image = np.asarray(image)
            augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (perlin_thr)
            # augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1 - msk) * image

            return Image.fromarray(np.uint8(augmented_image)), 1

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')

        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        a_img, label = self.augment_image(img, self.anomaly_source_paths[anomaly_source_idx])

        img = self.transform(img)
        a_img = self.strong_transform(a_img)

        assert img.size()[1:] == a_img.size()[1:], "image.size != a_img.size !!!"

        return img, a_img, label


class MVTecSimplexDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform

        self.simplexNoise = Simplex_CLASS()
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img_normal = self.transform(img)

        if random.random() > 0.5:
            return img_normal, img_normal
        ## simplex_noise
        size = 256
        img = img.resize((size, size))
        img = np.asarray(img)
        h_noise = np.random.randint(10, int(size // 8))
        w_noise = np.random.randint(10, int(size // 8))
        start_h_noise = np.random.randint(1, size - h_noise)
        start_w_noise = np.random.randint(1, size - w_noise)
        noise_size = (h_noise, w_noise)
        simplex_noise = self.simplexNoise.rand_3d_octaves((3, *noise_size), 6, 0.6)
        init_zero = np.zeros((256, 256, 3))
        init_zero[start_h_noise: start_h_noise + h_noise, start_w_noise: start_w_noise + w_noise,
        :] = 0.2 * simplex_noise.transpose(1, 2, 0)
        img_noise = img + init_zero * 255
        img_noise = Image.fromarray(np.uint8(img_noise))
        img_noise = self.transform(img_noise)

        return img_normal, img_noise
