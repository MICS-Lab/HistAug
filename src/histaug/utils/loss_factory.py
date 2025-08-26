import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarityLoss(nn.Module):
    """
    Cosine-based loss using F.cosine_similarity.
    If as_distance=True, returns 1 - cos(sim); else returns -cos(sim).
    """

    def __init__(
        self,
        reduction: str = "mean",
        dim: int = -1,
        eps: float = 1e-8,
        as_distance: bool = True,
    ):
        super().__init__()
        assert reduction in {"none", "mean", "sum"}
        self.reduction = reduction
        self.dim = dim
        self.eps = eps
        self.as_distance = as_distance

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cos = F.cosine_similarity(input, target, dim=self.dim, eps=self.eps)
        loss = (1.0 - cos) if self.as_distance else (-cos)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss  # "none"


CUSTOM_LOSSES = {
    "CosineSimilarityLoss": CosineSimilarityLoss,
}


class CombinedLoss(nn.Module):
    def __init__(self, losses: list[nn.Module], weights: list[float]) -> None:
        super().__init__()
        if len(losses) != len(weights):
            raise ValueError("Number of losses and weights must match")
        self.losses = nn.ModuleList(losses)
        self.weights = weights

    def forward(self, input_tensor, target):
        total_loss = torch.zeros(
            (), dtype=input_tensor.dtype, device=input_tensor.device
        )
        for loss_module, weight in zip(self.losses, self.weights):
            total_loss = total_loss + weight * loss_module(input_tensor, target)
        return total_loss


def _resolve_loss(name: str):
    if hasattr(nn, name):
        return getattr(nn, name)
    if name in CUSTOM_LOSSES:
        return CUSTOM_LOSSES[name]
    raise ValueError(
        f"Invalid loss: neither torch.nn.{name} nor custom '{name}' exists"
    )


def create_loss(loss_config: dict) -> nn.Module:
    base_losses = loss_config.get("base_loss")
    loss_weights = loss_config.get("loss_weights")

    if isinstance(base_losses, (list, tuple)):
        num_losses = len(base_losses)
        if (
            not isinstance(loss_weights, (list, tuple))
            or len(loss_weights) != num_losses
        ):
            original = loss_weights if loss_weights is not None else []
            loss_weights = [1.0] * num_losses
            print(
                f"[Warning] Provided loss_weights {original} does not match number of losses {num_losses}."
                f" Using default weights {loss_weights} for losses {base_losses}."
            )
        loss_modules = []
        for name in base_losses:
            loss_cls = _resolve_loss(name)
            loss_modules.append(loss_cls())
        return CombinedLoss(loss_modules, loss_weights)

    if isinstance(base_losses, str):
        name = base_losses
        if loss_weights is not None:
            if isinstance(loss_weights, (list, tuple)) and len(loss_weights) == 1:
                print(
                    f"[Info] Single loss '{name}' with weight {loss_weights[0]}. Weight list is redundant and will be ignored."
                )
            else:
                raise ValueError(
                    "For a single loss, loss_weights must be a single-element list if provided"
                )
        loss_cls = _resolve_loss(name)
        return loss_cls()

    raise ValueError(
        "`base_loss` must be a string or list of strings specifying loss names"
    )
