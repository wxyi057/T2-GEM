import torch
import torch.nn as nn
from torch import Tensor

from monai.losses import PerceptualLoss, PatchAdversarialLoss

class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z_mu: Tensor, z_sigma: Tensor) -> Tensor:
        kl_loss = 0.5 * torch.sum(
            z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4]
        )
        return torch.sum(kl_loss) / kl_loss.shape[0]

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

class PerceptualWithDiscriminator(nn.Module):
    def __init__(self,
                 kl_weight: float = 1.0,
                 perceptual_weight: float = 0.0,
                 disc_weight: float = 0.0,
                 spatial_dims: int = 3,
                 disc_start: int = 10000,
                 ):

        super().__init__()
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight
        self.disc_weight = disc_weight
        self.disc_start = disc_start

        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss(
            spatial_dims=spatial_dims,
            network_type="squeeze",
            fake_3d_ratio=0.25,
        )
        self.disc_loss = PatchAdversarialLoss(criterion="least_squares")
        self.kl_loss = KLDivergenceLoss()

    def _final_logit(self, logits):
        if isinstance(logits, (list, tuple)) and len(logits) > 0:
            return logits[-1]
        return logits

    def forward(
        self, 
        inputs, 
        reconstructions, 
        z_mu, 
        z_sigma, 
        optimizer_idx, 
        global_step, 
        logits_fake=None, 
        logits_real=None, 
        split="train"
    ):
        if optimizer_idx == 0:
            rec_loss = self.l1_loss(reconstructions.float(), inputs.float())

            p_val = self.perceptual_loss(reconstructions.float(), inputs.float())
            p_loss = self.perceptual_weight * p_val

            kld_val = self.kl_loss(z_mu, z_sigma)
            kld_loss = self.kl_weight * kld_val

            logits_fake = self._final_logit(logits_fake)
            disc_val = self.disc_loss(logits_fake, target_is_real=True, for_discriminator=False)

            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.disc_start)            
            loss = rec_loss + kld_loss + p_loss + disc_weight * disc_val

            log = {
                f"{split}/total_loss_0": loss.detach().mean().item(),
                f"{split}/rec_loss_0": rec_loss.detach().mean().item(),
                f"{split}/perceptual_loss_0": p_val.detach().mean().item(),
                f"{split}/kld_loss_0": kld_val.detach().mean().item(),
            }
            return loss, log

        if optimizer_idx == 1:
            logits_fake = self._final_logit(logits_fake)
            logits_real = self._final_logit(logits_real)

            d_loss_fake = self.disc_loss(logits_fake, target_is_real=False, for_discriminator=True)
            d_loss_real = self.disc_loss(logits_real, target_is_real=True, for_discriminator=True)
            d_loss = 0.5 * (d_loss_fake + d_loss_real)

            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.disc_start)
            loss = disc_weight * d_loss

            log = {
                f"{split}/d_loss_1": d_loss.detach().mean().item(),
                f"{split}/logits_real_1": logits_real.detach().mean().item(),
                f"{split}/logits_fake_1": logits_fake.detach().mean().item(),
            }
            return loss, log
