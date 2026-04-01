from torch.utils import data
import numpy as np
from monai import transforms
from pathlib import Path
import pandas as pd
import SimpleITK as sitk

def get_transform(image_keys, image_size):
    image_keys = tuple(image_keys)
    transforms_fn = [
        transforms.EnsureTyped(
            keys=image_keys,
            dtype=np.float32,
            allow_missing_keys=True,
        ),
        transforms.ScaleIntensityRangePercentilesd(
            keys=image_keys,
            lower=0,
            upper=99.5,
            b_min=-1.0,
            b_max=1.0,
            allow_missing_keys=True,
            clip=True
        ),
        transforms.CenterSpatialCropd(
            keys=image_keys,
            roi_size=tuple(image_size),
            allow_missing_keys=True,
        ),
        transforms.SpatialPadd(
            keys=image_keys,
            spatial_size=tuple(image_size),
        )
    ]
    return transforms.Compose(transforms_fn)

class ProstateDataset(data.Dataset):
    def __init__(
        self,
        root_dir,
        split,
        image_types,
        image_size,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_types = list(image_types)
        self.transform = get_transform(self.image_types, image_size)
        self.samples = self._build_split_samples()

    def _build_split_samples(self):
        meta_df = pd.read_csv(self.root_dir / 'metadata.csv')
        split_df = meta_df[meta_df["split"] == self.split].reset_index(drop=True)
        samples = []
        for _, row in split_df.iterrows():
            patient_dir = self.root_dir / row['id']
            for image_type in self.image_types:
                image_path = patient_dir / f'{image_type}.nii.gz'
                if image_path.exists():
                    samples.append({"path": image_path, "image_type": image_type})
        return samples

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image_path = sample["path"]
        image_type = sample["image_type"]
        image_array = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))[None, ...]
        data_dict = {
            image_type: image_array,
        }
        output_dict = self.transform(data_dict)
        return {
            "image": output_dict[image_type],
            "image_type": image_type,
        }

    def __len__(self):
        return len(self.samples)
