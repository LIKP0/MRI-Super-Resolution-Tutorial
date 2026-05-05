import torch
import pytorch_lightning as pl
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel
from torchmetrics.regression import MeanAbsoluteError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from loss import build_loss


class DDPM(pl.LightningModule):
    def __init__(
            self,
            image_size=224,  # Spatial size used by UNet2DModel.
            in_channels=2,  # Concatenated channels of noisy residual and condition.
            out_channels=1,  # Number of residual channels to generate.
            unet_channels=(64, 128, 256, 256),  # UNet block feature widths.
            num_train_timesteps=1000,  # Number of diffusion train timesteps.
            num_inference_steps=100,  # Default sampling steps for validation.
            initial_lr=1e-4,  # Initial learning rate for AdamW.
            val_metric_batches: int = 10,  # Batches used for sampled metrics.
    ):
        """Initialize the diffusion model for residual SR."""
        super().__init__()
        self.save_hyperparameters()

        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.unet_channels = unet_channels
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.learning_rate = initial_lr
        self.val_metric_batches = val_metric_batches

        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            beta_end=0.012,
            prediction_type="epsilon",
            clip_sample=True,
        )
        self.val_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            prediction_type="epsilon",
            clip_sample=True
        )

        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            block_out_channels=unet_channels,
            attention_head_dim=8,
            norm_num_groups=8,
            dropout=0.0,
            layers_per_block=2
        )
        self.criterion = build_loss("MSELoss")
        self.val_mae = MeanAbsoluteError()
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    @staticmethod
    def _reconstruct_from_src(src, residual):
        """Reconstruct output by adding residual to the center source channel block."""
        c = residual.size(1)
        start = (src.size(1) - c) // 2
        src_center = src[:, start:start + c]
        return src_center + residual

    @staticmethod
    def _to_minus1_1(x) -> torch.Tensor:
        """Map image range from [0, 1] to [-1, 1]."""
        return x * 2.0 - 1.0

    @staticmethod
    def _to_0_1(x) -> torch.Tensor:
        """Map image range from [-1, 1] to [0, 1]."""
        return (x + 1.0) / 2.0

    def forward(self, noisy_residual, timesteps, src):
        """Predict diffusion noise for the noisy residual input."""
        return self._predict_noise(noisy_residual, timesteps, src)

    def _predict_noise(self, noisy_residual, timesteps, src):
        """Run the conditional UNet to predict epsilon."""
        x = torch.cat([noisy_residual, src], dim=1)  # Conditioned UNet input.
        return self.unet(x, timesteps).sample

    def _compute_noise_loss(self, noise_pred, noise, timesteps):
        """Compute training loss for epsilon prediction."""
        return self.criterion(noise_pred, noise)

    def training_step(self, batch, batch_idx):
        """Run one diffusion training step on residual targets."""
        src_01, residual = batch["src"], batch["residual"]
        src = self._to_minus1_1(src_01)
        timesteps = torch.randint(0, self.num_train_timesteps, (residual.size(0),), device=residual.device).long()  # Random times.
        noise = torch.randn_like(residual)  # Gaussian noise target.
        noisy_residual = self.train_scheduler.add_noise(residual, noise, timesteps)  # Forward diffusion.
        noise_pred = self._predict_noise(noisy_residual, timesteps, src)  # Predict epsilon.
        loss = self._compute_noise_loss(noise_pred, noise, timesteps)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=src.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        """Run one validation step and update sampled-image metrics."""
        src_01, dst_01, residual = batch["src"], batch["dst"], batch["residual"]
        src = self._to_minus1_1(src_01)
        timesteps = torch.randint(0, self.num_train_timesteps, (residual.size(0),), device=residual.device).long()
        noise = torch.randn_like(residual)
        noisy_residual = self.train_scheduler.add_noise(residual, noise, timesteps)
        noise_pred = self._predict_noise(noisy_residual, timesteps, src)
        val_loss = self._compute_noise_loss(noise_pred, noise, timesteps)
        self.log("val/loss", val_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=src.size(0))

        if self.global_rank == 0 and batch_idx < self.val_metric_batches:
            generated_dst = self.sample(src_01, self.num_inference_steps)
            self.val_mae(generated_dst, dst_01)
            self.val_psnr(generated_dst, dst_01)
            self.val_ssim(generated_dst, dst_01)

            suffix = f"b{self.val_metric_batches}"
            log_dict = {
                f"val/mae_{suffix}": self.val_mae,
                f"val/psnr_{suffix}": self.val_psnr,
                f"val/ssim_{suffix}": self.val_ssim,
            }
            self.log_dict(log_dict, on_step=False, on_epoch=True, sync_dist=False, batch_size=src.size(0))

    def sample(self, src, num_inference_steps=50):
        """Sample residual with DDIM and reconstruct SR image."""
        src_01 = src
        src = self._to_minus1_1(src_01)
        b, _, h, w = src.shape
        device = src.device
        self.val_scheduler.set_timesteps(num_inference_steps, device=device)

        noisy_residual = self.val_scheduler.init_noise_sigma * torch.randn(  # Initial sampling noise.
            (b, self.out_channels, h, w),
            device=device,
            dtype=src.dtype
        )

        with torch.no_grad():
            for t in self.val_scheduler.timesteps:
                t_batch = t.expand(b).to(device=device, dtype=torch.long)
                noise_pred = self._predict_noise(noisy_residual, t_batch, src)
                step_out = self.val_scheduler.step(noise_pred, t, noisy_residual, eta=0.0)
                noisy_residual = step_out.prev_sample

        generated_01 = self._reconstruct_from_src(src_01, noisy_residual)  # SR = center source + sampled residual.
        return torch.clamp(generated_01, 0.0, 1.0)

    def sample_ddpm(self, src, num_inference_steps=1000):
        """Sample residual with DDPM and reconstruct SR image."""
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps

        src_01 = src
        src = self._to_minus1_1(src_01)
        b, _, h, w = src.shape
        device = src.device
        self.train_scheduler.set_timesteps(num_inference_steps, device=device)

        noisy_residual = self.train_scheduler.init_noise_sigma * torch.randn(  # Initial sampling noise.
            (b, self.out_channels, h, w),
            device=device,
            dtype=src.dtype
        )

        with torch.no_grad():
            for t in self.train_scheduler.timesteps:
                t_batch = t.expand(b).to(device=device, dtype=torch.long)
                noise_pred = self._predict_noise(noisy_residual, t_batch, src)
                step_out = self.train_scheduler.step(noise_pred, t, noisy_residual)
                noisy_residual = step_out.prev_sample

        generated_01 = self._reconstruct_from_src(src_01, noisy_residual)  # SR = center source + sampled residual.
        return torch.clamp(generated_01, 0.0, 1.0)

    def configure_optimizers(self):
        """Configure AdamW with cosine decay."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=total_steps,
            eta_min=1e-7,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            },
        }
