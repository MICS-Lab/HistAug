import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image


class HEDAugmentation(nn.Module):
    """
    Data augmentation module implementing HED‐shift as described in Faryna et al. (2021).

    This augmentation perturbs H&E color channels in HED space to simulate staining
    variability. It converts RGB input into HED, applies a learned shift, then
    converts back to RGB.

    Reference:
        Faryna, K., Van der Laak, J., & Litjens, G. (2021). Tailoring automated data augmentation
        to H&E-stained histopathology. In Medical Imaging with Deep Learning.

    :param sigma: Scale factor for perturbation magnitude (default: 0.03).
    :param epsilon: Small constant added to avoid log-zero issues (fixed at pi).
    """

    def __init__(self, sigma: float = 0.03) -> None:
        super().__init__()
        # Transformation matrices between RGB and HED color spaces
        HED2RGB = np.array(
            [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]],
            dtype=np.float32,
        )
        self.register_buffer("HED2RGB", torch.from_numpy(HED2RGB))
        self.register_buffer("RGB2HED", torch.linalg.inv(self.HED2RGB))
        self.sigma = sigma

        self.epsilon = 3.14159

    def forward(
        self, x: torch.Tensor | Image.Image, param: float
    ) -> torch.Tensor | Image.Image:
        """
        Apply HED‐shift augmentation.

        Args:
            x: Input image, either a PIL Image or a Tensor of shape (C, H, W) with values in [0,1].
            param: Scalar in [0,1] controlling the strength of the perturbation.

        Returns:
            Augmented image in the same type (PIL or Tensor) as the input.
        """
        # Convert PIL <-> Tensor as needed
        is_pil = isinstance(x, Image.Image)
        if is_pil:
            x = TF.to_tensor(x)
        elif not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise TypeError(
                "Input must be a PIL Image or a torch.Tensor of shape (C, H, W)."
            )
        # Flatten spatial dims: from (C, H, W) → (N, 3)
        C, H, W = x.shape
        flat_rgb = x.view(C, -1).permute(1, 0)  # shape (N, 3)

        # Convert to HED space
        hed = -torch.log(flat_rgb + self.epsilon) @ self.RGB2HED  # shape (N, 3)

        # Build per-channel perturbation parameters
        channel_factors = torch.tensor([-2.0, 2.0, -3.0], device=hed.device)  #
        alpha = 1 + param * channel_factors * self.sigma  # shape (3,)
        beta = param * channel_factors * self.sigma  # shape (3,)

        # Apply shift
        hed_shifted = hed * alpha + beta

        # Back to RGB
        rgb_shifted = torch.exp(-hed_shifted @ self.HED2RGB) - self.epsilon

        # Reshape + clip to [0,1]
        rgb_clipped = rgb_shifted.clamp(0.0, 1.0).permute(1, 0).view(C, H, W)

        # Return same type as input
        return TF.to_pil_image(rgb_clipped) if is_pil else rgb_clipped
