import torch
import pytorch_lightning as pl
from diffusers import UNet2DModel
from torch import nn
from torchmetrics.regression import MeanAbsoluteError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


class UNet(pl.LightningModule):
    def __init__(
            self,
            image_size=224,  # Spatial size used by UNet2DModel.
            in_channels=1,  # Number of source input channels.
            out_channels=1,  # Number of predicted residual channels.
            unet_channels=(64, 128, 256, 256),  # UNet feature widths per stage.
            initial_lr=1e-4,  # Initial learning rate for AdamW.
    ):
        """Initialize the residual SR UNet model and metric trackers."""
        super().__init__()
        self.save_hyperparameters()

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

        self.initial_lr = initial_lr
        self.criterion = nn.L1Loss()

        self.val_mae = MeanAbsoluteError()
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.test_mae = MeanAbsoluteError()
        self.test_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.test_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    @staticmethod
    def _reconstruct_from_src(src, residual):
        """Reconstruct output by adding residual to the center source channel block."""
        c = residual.size(1)
        start = (src.size(1) - c) // 2
        src_center = src[:, start:start + c]
        return src_center + residual

    def forward(self, x):
        """Predict residual from source input."""
        b = x.size(0)
        timesteps = torch.zeros((b,), device=x.device, dtype=torch.long)
        return self.unet(x, timesteps).sample

    def _shared_step(self, batch):
        """Compute residual loss and reconstructed prediction for one batch."""
        src, dst = batch["src"], batch["dst"]
        residual_pred = self(src)  # Predict residual from source.
        outputs = self._reconstruct_from_src(src, residual_pred)  # Reconstruct SR image.
        # For UNet supervision, optimize directly on reconstructed image.
        loss = self.criterion(outputs, dst)
        return outputs, dst, loss

    def sample(self, src, num_inference_steps=None):
        """Predict SR image for inference."""
        residual_pred = self(src)
        outputs = self._reconstruct_from_src(src, residual_pred)
        return torch.clamp(outputs, 0.0, 1.0)

    def training_step(self, batch, batch_idx):
        """Run one training step and log training loss."""
        _, _, loss = self._shared_step(batch)
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch["dst"].size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Run one validation step and update image quality metrics."""
        y_hat, y, loss = self._shared_step(batch)
        y_hat = torch.clamp(y_hat, 0.0, 1.0)
        self.val_mae(y_hat, y)
        self.val_psnr(y_hat, y)
        self.val_ssim(y_hat, y)
        self.log_dict(
            {"val/loss": loss, "val/mae": self.val_mae, "val/psnr": self.val_psnr, "val/ssim": self.val_ssim},
            on_step=False, on_epoch=True, sync_dist=True, batch_size=y.size(0)
        )

    def configure_optimizers(self):
        """Configure AdamW and cosine learning-rate schedule."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.initial_lr)
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=total_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            },
        }
