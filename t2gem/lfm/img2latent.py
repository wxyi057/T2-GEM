import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from monai import transforms
from monai.networks.nets import AutoencoderKL
import hydra
from omegaconf import DictConfig
from t2gem.utils.device import get_amp_dtype_and_device
import SimpleITK as sitk

class RunningStats:
    def __init__(self):
        self.count = 0
        self.sum = torch.zeros((), dtype=torch.float64)
        self.sum_sq = torch.zeros((), dtype=torch.float64)

    def update(self, tensor: torch.Tensor) -> None:
        flattened = tensor.detach().to(dtype=torch.float64).flatten()
        self.count += flattened.numel()
        self.sum += flattened.sum()
        self.sum_sq += (flattened * flattened).sum()

    def std(self):
        if self.count < 2:
            return None
        mean = self.sum / self.count
        var = (self.sum_sq - self.count * mean * mean) / (self.count - 1)
        var = torch.clamp(var, min=0.0)
        return torch.sqrt(var)

    def scale_factor(self):
        std = self.std()
        if std is None:
            return None
        if std.item() == 0:
            return None
        return 1.0 / std

def init_autoencoder(config, ckpt):
    autoencoderkl = AutoencoderKL(
        spatial_dims=config.autoencoder.spatial_dims,
        in_channels=config.autoencoder.in_channels,
        out_channels=config.autoencoder.out_channels,
        num_res_blocks=config.autoencoder.num_res_blocks,
        channels=config.autoencoder.num_channels,
        attention_levels=config.autoencoder.attention_levels,
        latent_channels=config.autoencoder.z_channels,
        norm_num_groups=config.autoencoder.norm_num_groups,
    )
    state_dict = torch.load(ckpt, map_location='cpu')['model']
    autoencoderkl.load_state_dict(state_dict)
    return autoencoderkl

@hydra.main(version_base=None, config_path="", config_name="config")
def main(config: DictConfig) -> None:
    transforms_fn = transforms.Compose([
        transforms.EnsureType(
            dtype=np.float32,
        ),
        transforms.ScaleIntensityRangePercentiles(
            lower=0,
            upper=99.5,
            b_min=-1.0,
            b_max=1.0,
            clip=True
        ),
        transforms.CenterSpatialCrop(
            roi_size=tuple(config.data.image_size),
        ),
        transforms.SpatialPad(
            spatial_size=tuple(config.data.image_size),
        )
    ])
    data_df = pd.read_csv(config.data.csv_path)

    _, device = get_amp_dtype_and_device()
    autoencoder_ckpt = config.autoencoder.ckpt
    autoencoder = init_autoencoder(config, autoencoder_ckpt).to(device)
    print(f"Loaded shared autoencoder for T2 and DWI from {autoencoder_ckpt}")
    autoencoder.eval()

    stats = RunningStats()

    print(f"Starting to process data")
    with torch.no_grad():
        for index, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Processing data"):
            t2_image_path = row['t2_image']
            dwi_image_path = row['dwi_image']
            t2_latent_path = row['t2_latent']
            dwi_latent_path = row['dwi_latent']
            t2_image_array = sitk.GetArrayFromImage(sitk.ReadImage(str(t2_image_path)))[None, ...]
            dwi_image_array = sitk.GetArrayFromImage(sitk.ReadImage(str(dwi_image_path)))[None, ...]
            t2_image_tensor = transforms_fn(t2_image_array).to(device)
            dwi_image_tensor = transforms_fn(dwi_image_array).to(device)
            t2_latent = autoencoder.encode_stage_2_inputs(t2_image_tensor.unsqueeze(0))
            dwi_latent = autoencoder.encode_stage_2_inputs(dwi_image_tensor.unsqueeze(0))
            t2_latent_cpu = t2_latent.detach().cpu()
            dwi_latent_cpu = dwi_latent.detach().cpu()
            stats.update(t2_latent_cpu)
            stats.update(dwi_latent_cpu)
            np.savez_compressed(t2_latent_path, data=t2_latent_cpu.squeeze(0).numpy())
            np.savez_compressed(dwi_latent_path, data=dwi_latent_cpu.squeeze(0).numpy())

    scale = stats.scale_factor()
    print(f"Combined dataset (T2 + DWI) std: {stats.std().item():.6f}, scale_factor (1/std): {scale.item():.6f}")

if __name__ == "__main__":
    main()
