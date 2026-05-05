import torch
import pytorch_lightning as pl
from diffusers import FlowMatchEulerDiscreteScheduler, UNet2DModel
from torchmetrics.regression import MeanAbsoluteError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from loss import build_loss


class FlowMatching(pl.LightningModule):
    def __init__(
            self,
            image_size=224,  # Spatial size used by UNet2DModel.
            in_channels=2,  # Concatenated channels of state and condition.
            out_channels=1,  # Number of residual channels to generate.
            unet_channels=(64, 128, 256, 256),  # UNet block feature widths.
            num_train_timesteps=1000,  # Number of scheduler train timesteps.
            num_inference_steps=100,  # Default sampling steps for validation.
            initial_lr=1e-4,  # Initial learning rate for AdamW.
            val_metric_batches: int = 10,  # Batches used for sampled metrics.
    ):
        """Initialize the flow-matching model for residual SR."""
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

        self.train_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.num_train_timesteps,
        )
        self.val_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.num_train_timesteps,
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
            layers_per_block=2,
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

    def forward(self, xt, timesteps, src):
        """Predict flow velocity for the current residual state."""
        return self._predict_velocity(xt, timesteps, src)

    def _predict_velocity(self, xt, timesteps, src):
        """Run the conditional UNet on concatenated state and source."""
        x = torch.cat([xt, src], dim=1)
        return self.unet(x, timesteps).sample

    def _sample_training_timesteps(self, batch_size, device):
        """Sample random timesteps used for flow training."""
        step_indices = torch.randint(0, self.train_scheduler.timesteps.numel(), (batch_size,), device=device)
        timesteps = self.train_scheduler.timesteps.to(device=device)[step_indices]
        return timesteps

    def _mix_flow_pair(self, residual, noise, timesteps):
        """Construct noisy state and target velocity for flow matching."""
        xt = self.train_scheduler.scale_noise(residual, timesteps, noise)  # Interpolated state.
        velocity = noise - residual  # Target flow vector.
        return xt, velocity

    def training_step(self, batch, batch_idx):
        """Compute flow loss on residual targets."""
        src_01, residual = batch["src"], batch["residual"]
        src = self._to_minus1_1(src_01)

        timesteps = self._sample_training_timesteps(residual.size(0), residual.device)  # Random times.
        noise = torch.randn_like(residual)  # Gaussian endpoint sample.
        xt, velocity = self._mix_flow_pair(residual, noise, timesteps)
        velocity_pred = self._predict_velocity(xt, timesteps, src)  # Predicted flow.
        loss = self.criterion(velocity_pred, velocity)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=src.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        """Compute validation flow loss and sampled image metrics."""
        src_01, dst_01, residual = batch["src"], batch["dst"], batch["residual"]
        src = self._to_minus1_1(src_01)

        timesteps = self._sample_training_timesteps(residual.size(0), residual.device)
        noise = torch.randn_like(residual)
        xt, velocity = self._mix_flow_pair(residual, noise, timesteps)
        velocity_pred = self._predict_velocity(xt, timesteps, src)
        val_loss = self.criterion(velocity_pred, velocity)
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

    def sample(self, src, num_inference_steps=8):
        """Sample residual and reconstruct SR image by adding source input."""
        src_01 = src
        src = self._to_minus1_1(src_01)
        b, _, h, w = src.shape
        device = src.device

        self.val_scheduler.set_timesteps(num_inference_steps, device=device)
        xt = torch.randn((b, self.out_channels, h, w), device=device, dtype=src.dtype)  # Start from noise.

        with torch.no_grad():
            for t in self.val_scheduler.timesteps:
                t_batch = t.expand(b).to(device=device, dtype=src.dtype)
                velocity_pred = self._predict_velocity(xt, t_batch, src)
                step_out = self.val_scheduler.step(velocity_pred, t, xt)
                xt = step_out.prev_sample

        generated_01 = self._reconstruct_from_src(src_01, xt)  # SR = center source + sampled residual.
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
