from pathlib import Path
import random
from collections import OrderedDict

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


def discover_cases(data_dir: str | Path, max_cases: int | None = None):
    """Return image/label pairs from an extracted MSD Task01 directory."""
    root = Path(data_dir)
    image_dir, label_dir = root / "imagesTr", root / "labelsTr"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(
            f"Expected imagesTr/ and labelsTr/ under {root}. See README data setup."
        )
    pairs = [(p, label_dir / p.name) for p in sorted(image_dir.glob("*.nii*"))]
    pairs = [(image, label) for image, label in pairs if label.exists()]
    if not pairs:
        raise FileNotFoundError(f"No paired NIfTI image and label files found under {root}")
    return pairs[:max_cases] if max_cases else pairs


def patient_split(pairs, seed=42, train=0.70, val=0.15):
    """Split whole patients before making 2D slices to prevent leakage."""
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    n_train = max(1, int(n * train))
    n_val = max(1, int(n * val)) if n >= 3 else 0
    if n_train + n_val >= n:
        n_train, n_val = max(1, n - 2), 1
    return pairs[:n_train], pairs[n_train:n_train + n_val], pairs[n_train + n_val:]


def load_case(image_path: Path, label_path: Path):
    """Read a 4-channel volume and binary whole-tumor mask, in axial orientation."""
    image = np.asarray(nib.load(str(image_path)).dataobj, dtype=np.float32)
    label = np.asarray(nib.load(str(label_path)).dataobj, dtype=np.uint8)
    if image.ndim != 4 or image.shape[-1] != 4:
        raise ValueError(f"Expected image shape [X,Y,Z,4], got {image.shape} ({image_path})")
    if label.shape != image.shape[:3]:
        raise ValueError(f"Image/mask shape mismatch: {image.shape} vs {label.shape}")
    # Per-modality z-score over nonzero brain voxels; robustly clip extreme values.
    for channel in range(4):
        volume = image[..., channel]
        brain = volume != 0
        if brain.any():
            values = volume[brain]
            mean, std = float(values.mean()), float(values.std())
            image[..., channel] = np.where(brain, np.clip((volume - mean) / max(std, 1e-6), -5, 5), 0)
    mask = (label > 0).astype(np.float32)
    return image, mask


class AxialSliceDataset(Dataset):
    """On-demand axial slices; sample all tumor slices and every Nth empty slice."""
    def __init__(self, pairs, negative_stride=4):
        self.pairs = list(pairs)
        self.index = []
        self._cache = OrderedDict()
        for case_id, (image_path, label_path) in enumerate(self.pairs):
            label = np.asarray(nib.load(str(label_path)).dataobj)
            for z in range(label.shape[2]):
                if np.any(label[:, :, z] > 0) or z % negative_stride == 0:
                    self.index.append((case_id, z))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        case_id, z = self.index[item]
        if case_id not in self._cache:
            self._cache[case_id] = load_case(*self.pairs[case_id])
            while len(self._cache) > 2:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(case_id)
        image, mask = self._cache[case_id]
        x = torch.from_numpy(np.ascontiguousarray(image[:, :, z, :].transpose(2, 0, 1)))
        y = torch.from_numpy(np.ascontiguousarray(mask[:, :, z][None, ...]))
        return x, y
