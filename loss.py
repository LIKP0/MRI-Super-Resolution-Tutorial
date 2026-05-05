from torch import nn


class MSELoss(nn.Module):
    """Wrap MSE loss for unified management and future extension."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(self, pred, target):
        return self.loss_fn(pred, target)


def build_loss(loss_name: str, **kwargs) -> nn.Module:
    """Build a loss function instance from its string class name."""
    if not isinstance(loss_name, str) or loss_name == "":
        raise ValueError("loss_name must be a non-empty string and strictly match the class name.")

    candidate = globals().get(loss_name, None)

    if candidate is None:
        available_losses = sorted(
            name for name, obj in globals().items()
            if isinstance(obj, type) and issubclass(obj, nn.Module) and name.endswith("Loss")
        )
        raise ValueError(
            f"Unsupported loss '{loss_name}'. "
            f"Available losses in loss.py: {available_losses}"
        )

    if not isinstance(candidate, type) or not issubclass(candidate, nn.Module):
        raise TypeError(f"'{loss_name}' is found but is not an nn.Module loss class.")

    return candidate(**kwargs)
