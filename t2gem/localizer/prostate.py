import math
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import SimpleITK as sitk
from monai.transforms import (
    Compose,
    EnsureTyped,
    RandAffined,
    RandBiasFieldd,
    ScaleIntensityRangePercentilesd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandFlipd,
)
from omegaconf import DictConfig
from typing import List, Dict, Any

from t2gem.localizer.utils import normalize_modalities

def get_transform(config: DictConfig, modalities: List[str] | None = None):
    img_keys = normalize_modalities(config) if modalities is None else [str(mod).lower() for mod in modalities]
    mask_keys = ["gland"]
    affine_mode = tuple(["bilinear"] * len(img_keys) + ["nearest"])

    train_transforms = [
        EnsureTyped(keys=img_keys + mask_keys, dtype=np.float32, allow_missing_keys=False),
        ScaleIntensityRangePercentilesd(
                keys=img_keys, lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
            ),
        RandAffined(
            keys=img_keys + mask_keys,
            prob=0.5,
            rotate_range=(np.pi/12, np.pi/12, np.pi/12), 
            scale_range=(0.1, 0.1, 0.1),
            mode=affine_mode,
            padding_mode="border"
        ),
        RandFlipd(keys=img_keys + mask_keys, prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys=img_keys, factors=0.1, prob=0.8),
        RandShiftIntensityd(keys=img_keys, offsets=0.1, prob=0.8),
        RandBiasFieldd(keys=img_keys, prob=0.2),
    ]

    val_transforms = [
        EnsureTyped(keys=img_keys + mask_keys, dtype=np.float32, allow_missing_keys=False),
        ScaleIntensityRangePercentilesd(
                keys=img_keys, lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
            ),
    ]

    return Compose(train_transforms), Compose(val_transforms)

class ProstateDataset(Dataset):
    def __init__(self, config, meta_df, is_train = False, modalities = None):
        super().__init__()
        self.config = config
        self.root_dir = Path(config.data.root_dir)
        self.data_info = meta_df
        self.modalities = normalize_modalities(config) if modalities is None else [str(mod).lower() for mod in modalities]
        self.roi_margin_ratio = float(config.data.roi_margin_ratio)
        self.is_train = bool(is_train)

        train_transform, val_transform = get_transform(config, self.modalities)
        self.transform = train_transform if is_train else val_transform
        
    def __len__(self) -> int:
        return len(self.data_info)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.data_info.iloc[index]
        patient_id = row["case_id"]

        gland_zone_path = self.root_dir / "masks" / "gland_zone" / f"{patient_id}.nii.gz"
        pz_tz_zone_path = self.root_dir / "masks" / "pz_tz_zone" / f"{patient_id}.nii.gz"
        zonal_label_tensor = self._build_labels(row)

        images = {}
        for mod in self.modalities:
            img_path = self.root_dir / "images" / mod / f"{patient_id}.nii.gz"
            images[mod] = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path))).astype(np.float32)[None, ...]
        gland_zone_raw = sitk.GetArrayFromImage(sitk.ReadImage(str(gland_zone_path))).astype(np.int16)
        gland_zone_array = gland_zone_raw.astype(np.float32)[None, ...]

        zone_regions_tensor: torch.Tensor | None = None
        zone_region_weights_tensor: torch.Tensor | None = None
        if not self.is_train:
            pz_tz_zone_raw = sitk.GetArrayFromImage(sitk.ReadImage(str(pz_tz_zone_path))).astype(np.int16)
            zone_region_weights = self._compute_zone_region_weights_from_pz_tz(
                gland_zone=gland_zone_raw, pz_tz_zone=pz_tz_zone_raw, patient_id=str(patient_id)
            )
            zone_region_weights_tensor = torch.as_tensor(zone_region_weights, dtype=torch.float32)

            zone_regions = self._infer_zone_regions_from_pz_tz(
                gland_zone=gland_zone_raw, pz_tz_zone=pz_tz_zone_raw, patient_id=str(patient_id)
            )
            zone_regions_tensor = torch.as_tensor(zone_regions, dtype=torch.long)

        data = {**images, 'gland': gland_zone_array}
        data = self.transform(data)

        stacked_image = torch.cat([data[mod] for mod in self.modalities], dim=0)
        
        rois = self._get_rois(data['gland'].squeeze(0))
        
        sample: Dict[str, Any] = {
            't2': stacked_image,
            'patient_id': patient_id, 
            'zonal_labels': zonal_label_tensor, 
            'rois': rois
        }
        if zone_regions_tensor is not None:
            sample["zone_regions"] = zone_regions_tensor
        if zone_region_weights_tensor is not None:
            sample["zone_region_weights"] = zone_region_weights_tensor
        return sample

    def _build_labels(self, row: pd.Series) -> torch.Tensor:
        labels: List[int] = []
        for i in range(1, 21):
            val = row[f"isup_zone_{i}"]
            if val <= 0:
                labels.append(0)
            elif val == 1:
                labels.append(1)
            elif val == 2:
                labels.append(2)
            else:
                labels.append(3)
        return torch.tensor(labels, dtype=torch.long)

    def _get_rois(self, gland_zone_array: torch.Tensor) -> torch.Tensor:
        if not isinstance(gland_zone_array, torch.Tensor):
            raise TypeError("gland_zone_array must be a torch.Tensor.")
        if gland_zone_array.ndim != 3:
            raise ValueError("gland_zone_array must be a 3D tensor with shape [Z, Y, X].")
        if self.roi_margin_ratio < 0:
            raise ValueError("roi_margin_ratio must be non-negative.")

        mask = gland_zone_array.detach().cpu()
        z_size, y_size, x_size = mask.shape
        bboxes: List[List[int]] = []

        for label_value in range(1, 21):
            z_indices, y_indices, x_indices = (mask == label_value).nonzero(as_tuple=True)
            if z_indices.numel() == 0:
                raise ValueError(f"Missing zone label {label_value} in gland_zone mask; expected 20 zones (1..20).")

            zmin_raw, zmax_raw = int(z_indices.min().item()), int(z_indices.max().item())
            ymin_raw, ymax_raw = int(y_indices.min().item()), int(y_indices.max().item())
            xmin_raw, xmax_raw = int(x_indices.min().item()), int(x_indices.max().item())

            zmin, zmax = self._expand_with_ratio(zmin_raw, zmax_raw, z_size)
            ymin, ymax = self._expand_with_ratio(ymin_raw, ymax_raw, y_size)
            xmin, xmax = self._expand_with_ratio(xmin_raw, xmax_raw, x_size)

            bboxes.append([zmin, ymin, xmin, zmax + 1, ymax + 1, xmax + 1])

        if len(bboxes) != 20:
            raise ValueError(f"Expected 20 ROIs (zones 1..20), got {len(bboxes)}.")

        return torch.as_tensor(bboxes, dtype=torch.float32)

    def _expand_with_ratio(self, start: int, end: int, max_size: int) -> tuple[int, int]:
        length = end - start + 1
        extra = int(math.ceil(length * self.roi_margin_ratio))
        new_start = max(start - extra, 0)
        new_end = min(end + extra, max_size - 1)
        return new_start, new_end

    def _compute_zone_region_weights_from_pz_tz(
        self,
        gland_zone: np.ndarray,
        pz_tz_zone: np.ndarray,
        patient_id: str,
    ) -> np.ndarray:
        gland = gland_zone.astype(np.int64, copy=False)
        gland_min = int(gland.min())
        gland_max = int(gland.max())
        if gland_min < 0 or gland_max > 20:
            raise ValueError(
                f"case_id={patient_id}: unexpected gland_zone label range [{gland_min}, {gland_max}]. "
                "Expected labels in [0..20] where 1..20 are Barzell zones."
            )
        reg = pz_tz_zone.astype(np.int64, copy=False)
        reg = np.where(reg == 2, 2, np.where(reg == 1, 1, 0))

        pair = gland * 3 + reg
        counts = np.bincount(pair.reshape(-1), minlength=(21 * 3)).reshape(21, 3)

        zone_region_weights = np.zeros((20, 2), dtype=np.float32)
        for zone_label in range(1, 21):
            pz_overlap = int(counts[zone_label, 1])
            tz_overlap = int(counts[zone_label, 2])
            denom = pz_overlap + tz_overlap
            if denom == 0:
                raise ValueError(
                    f"case_id={patient_id}: zone {zone_label} has zero overlap with both PZ(label=1) and TZ(label=2). "
                )
            zone_region_weights[zone_label - 1, 0] = float(pz_overlap) / float(denom)
            zone_region_weights[zone_label - 1, 1] = float(tz_overlap) / float(denom)

        return zone_region_weights

    def _infer_zone_regions_from_pz_tz(
        self,
        gland_zone: np.ndarray,
        pz_tz_zone: np.ndarray,
        patient_id: str,
    ) -> np.ndarray:
        weights = self._compute_zone_region_weights_from_pz_tz(
            gland_zone=gland_zone, pz_tz_zone=pz_tz_zone, patient_id=patient_id
        )
        zone_regions = np.where(weights[:, 0] >= weights[:, 1], 1, 2).astype(np.int64, copy=False)
        return zone_regions
