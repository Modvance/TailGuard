"""Datasets used by the TailGuard MVTec-compatible training protocol."""

import glob
import os

import numpy as np
import torch
import torch.multiprocessing
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms


torch.multiprocessing.set_sharing_strategy("file_system")


def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.CenterCrop(isize),
            transforms.Normalize(mean=mean_train, std=std_train),
        ]
    )
    gt_transforms = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.CenterCrop(isize),
            transforms.ToTensor(),
        ]
    )
    return data_transforms, gt_transforms


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
        rel_path = os.path.relpath(img_path, self.data_root).replace("\\", "/")
        return {
            "img_path": img_path,
            "rel_path": rel_path,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "base_idx": int(base_idx),
        }


class TrainDiagDataset(_BaseImageFolderMetaDataset):
    def __init__(
        self,
        dataset,
        data_root,
        class_name,
        class_id,
        sample_offset=0,
        contaminated_paths=None,
    ):
        super().__init__(dataset, data_root, class_name, class_id)
        self.sample_offset = sample_offset
        self.contaminated_paths = contaminated_paths

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        meta = self._build_base_meta(idx)
        is_contaminated = None
        if self.contaminated_paths is not None:
            is_contaminated = int(meta["rel_path"] in self.contaminated_paths)
        meta.update(
            {
                "sample_idx": self.sample_offset + idx,
                "is_contaminated": -1 if is_contaminated is None else is_contaminated,
            }
        )
        meta.pop("rel_path", None)
        return image, label, meta


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == "train":
            self.img_path = os.path.join(root, "train")
        else:
            self.img_path = os.path.join(root, "test")
            self.gt_path = os.path.join(root, "ground_truth")
        self.transform = transform
        self.gt_transform = gt_transform
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()
        self.cls_idx = 0

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []
        for defect_type in os.listdir(self.img_path):
            patterns = ("*.png", "*.JPG", "*.jpg", "*.jpeg", "*.bmp")
            img_paths = []
            for pattern in patterns:
                img_paths.extend(glob.glob(os.path.join(self.img_path, defect_type, pattern)))
            img_paths.sort()
            if defect_type == "good":
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(["good"] * len(img_paths))
            else:
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type, "*.png"))
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))
        if len(img_tot_paths) != len(gt_tot_paths):
            raise ValueError("test images and ground-truth masks are not paired")
        return (
            np.asarray(img_tot_paths),
            np.asarray(gt_tot_paths),
            np.asarray(tot_labels),
            np.asarray(tot_types),
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        gt = self.gt_paths[idx]
        label = self.labels[idx]
        image = self.transform(Image.open(img_path).convert("RGB"))
        if label == 0:
            mask = torch.zeros([1, image.size(-2), image.size(-1)])
        else:
            mask = self.gt_transform(Image.open(gt))
        if image.size()[1:] != mask.size()[1:]:
            raise ValueError("image and ground-truth mask sizes differ")
        return image, mask, label, img_path
