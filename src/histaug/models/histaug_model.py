import math
import random
from collections import OrderedDict
from typing import Literal

import torch
from timm.layers import DropPath, Mlp
from timm.models.vision_transformer import LayerScale
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


class Attention(nn.Module):
    """
    Multi-head attention module with optional query/key normalization.

    :param dim: Total feature dimension.
    :param num_heads: Number of attention heads.
    :param qkv_bias: Whether to include bias terms in linear projections.
    :param qk_norm: Whether to apply LayerNorm to individual head queries and keys.
    :param attn_drop: Dropout probability for attention weights.
    :param proj_drop: Dropout probability after the output projection.
    :param norm_layer: Normalization layer to use if qk_norm is True.

    :return: Output tensor of shape (B, N1, dim) after attention and projection.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-head attention.

        :param x: Query tensor of shape (B, N1, dim).
        :param z: Key/Value tensor of shape (B, N2, dim).
        :return: Attention output tensor of shape (B, N1, dim).
        """
        B, N1, C = x.shape
        B, N2, C = z.shape

        q = self.q(x).reshape([B, N1, self.num_heads, self.head_dim]).swapaxes(1, 2)
        kv = (
            self.kv(z)
            .reshape(B, N2, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)
        with sdpa_kernel(
            [
                SDPBackend.MATH,
            ]
        ):
            x = F.scaled_dot_product_attention(
                query=q, key=k, value=v, dropout_p=self.attn_drop, scale=self.scale
            )

        x = x.transpose(1, 2).reshape(B, N1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: float = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
    ) -> None:
        """
        Transformer block combining attention and MLP with residual connections and optional LayerScale and DropPath.

        :param dim: Feature dimension.
        :param num_heads: Number of attention heads.
        :param mlp_ratio: Ratio for hidden dimension in MLP.
        :param qkv_bias: Whether to include bias in QKV projections.
        :param qk_norm: Whether to normalize Q and K.
        :param proj_drop: Dropout probability after output projection.
        :param attn_drop: Dropout probability for attention.
        :param init_values: Initial value for LayerScale (if None, LayerScale is Identity).
        :param drop_path: Dropout probability for stochastic depth.
        :param act_layer: Activation layer for MLP.
        :param norm_layer: Normalization layer.
        :param mlp_layer: MLP module class.

        :return: Output tensor of shape (B, N, dim).
        """
        super().__init__()
        self.x_norm = nn.LayerNorm(dim)
        self.z_norm = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a transformer block.

        :param x: Input tensor of shape (B, N, dim).
        :param z: Conditioning tensor for attention of same shape.
        :return: Output tensor of same shape after attention and MLP.
        """
        x = x + self.drop_path1(self.ls1(self.attn(self.x_norm(x), self.z_norm(z))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class HistaugModel(nn.Module):
    """
    Hierarchical augmentation transformer model for embedding input features and augmentations.

    :param input_dim: Dimensionality of raw input features.
    :param depth: Number of transformer blocks.
    :param num_heads: Number of attention heads.
    :param mlp_ratio: Ratio for hidden features in MLP layers.
    :param use_transform_pos_embeddings: Whether to include sequence positional embeddings for augmentations.
    :param positional_encoding_type: Type for transform positional embeddings ('learnable' or 'sinusoidal').
    :param final_activation: Name of activation layer for final head.
    :param chunk_size: Number of chunks to split the input.
    :param transforms: Dictionary containing augmentation parameter configurations.
    :param kwargs: Additional unused keyword arguments.

    :return: Output tensor of shape (B, input_dim) after augmentation and transformer processing.
    """

    def __init__(
        self,
        input_dim,
        depth,
        num_heads,
        mlp_ratio,
        use_transform_pos_embeddings=True,
        positional_encoding_type="learnable",  # New parameter
        final_activation="Identity",
        chunk_size=16,
        transforms=None,
        **kwargs,
    ):
        super().__init__()
        # Features embedding
        assert input_dim % chunk_size == 0, "input_dim must be divisble by chunk_size"

        self.input_dim = input_dim

        self.chunk_size = chunk_size
        self.transforms_parameters = transforms["parameters"]
        self.aug_param_names = sorted(self.transforms_parameters.keys())

        self.use_transform_pos_embeddings = use_transform_pos_embeddings
        self.positional_encoding_type = (
            positional_encoding_type  # Store the new parameter
        )
        self.embed_dim = self.input_dim // self.chunk_size
        self.chunk_pos_embeddings = self._get_sinusoidal_embeddings(
            self.chunk_size, self.embed_dim
        )
        self.register_buffer("chunk_pos_embeddings_buffer", self.chunk_pos_embeddings)
        if use_transform_pos_embeddings:
            if positional_encoding_type == "learnable":
                self.sequence_pos_embedding = nn.Embedding(
                    len(transforms["parameters"]), self.embed_dim
                )
            elif positional_encoding_type == "sinusoidal":
                sinusoidal_embeddings = self._get_sinusoidal_embeddings(
                    len(transforms["parameters"]), self.embed_dim
                )
                self.register_buffer("sequence_pos_embedding", sinusoidal_embeddings)
            else:
                raise ValueError(
                    f"Invalid positional_encoding_type: {positional_encoding_type}. Choose 'learnable' or 'sinusoidal'."
                )
        else:
            print("Do not use transform positional embeddings")

        self.transform_embeddings = self._get_transforms_embeddings(
            transforms["parameters"], self.embed_dim
        )

        self.features_embed = nn.Sequential(
            nn.Linear(input_dim, self.embed_dim), nn.LayerNorm(self.embed_dim)
        )

        self.blocks = nn.ModuleList(
            [
                Block(dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(self.embed_dim)

        if hasattr(nn, final_activation):
            self.final_activation = getattr(nn, final_activation)()
        else:
            raise ValueError(f"Activation {final_activation} is not found in torch.nn")

        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim), self.final_activation
        )

    def _get_sinusoidal_embeddings(self, num_positions, embed_dim):
        """
        Create sinusoidal embeddings for positional encoding.

        :param num_positions: Number of positions to encode.
        :param embed_dim: Dimensionality of each embedding vector.
        :return: Tensor of shape (num_positions, embed_dim) containing positional encodings.
        """
        assert embed_dim % 2 == 0, "embed_dim must be even"
        position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / embed_dim)
        )  # (embed_dim/2)

        pe = torch.zeros(num_positions, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def _get_transforms_embeddings(self, transforms, embed_dim):
        """
        Create embedding modules for each augmentation parameter.

        :param transforms: Mapping of augmentation names to configuration.
        :param embed_dim: Dimensionality of the embeddings.
        :return: ModuleDict of embeddings for each augmentation type.
        """
        transform_embeddings = nn.ModuleDict()
        for aug_name in transforms:
            if aug_name in [
                "rotation",
                "h_flip",
                "v_flip",
                "gaussian_blur",
                "erosion",
                "dilation",
            ]:
                # Discrete transformations
                transform_embeddings[aug_name] = nn.Embedding(
                    num_embeddings=2 if aug_name != "rotation" else 4,
                    embedding_dim=embed_dim,
                )
            elif aug_name in ["crop"]:
                # Discrete transformations
                transform_embeddings[aug_name] = nn.Embedding(
                    num_embeddings=6, embedding_dim=embed_dim
                )
            elif aug_name in [
                "brightness",
                "contrast",
                "saturation",
                "hed",
                "hue",
                "gamma",
            ]:
                # Continuous transformations
                transform_embeddings[aug_name] = nn.Sequential(
                    nn.Linear(1, embed_dim * 2),
                    nn.SiLU(),
                    nn.Linear(embed_dim * 2, embed_dim),
                )
            else:
                raise ValueError(
                    f"{aug_name} is not a valid augmentation parameter name"
                )
        return transform_embeddings

    def forward_aug_params_embed(self, aug_params):
        """
        Embed augmentation parameters and add positional embeddings if enabled.

        :param aug_params: OrderedDict mapping augmentation names to (value_tensor, position_tensor).
        :return: Tensor of shape (B, K, embed_dim) of embedded transform tokens.
        """
        z_transforms = []
        for aug_name, (aug_param, pos) in aug_params.items():
            if aug_name in [
                "rotation",
                "h_flip",
                "v_flip",
                "gaussian_blur",
                "erosion",
                "dilation",
                "crop",
            ]:
                z_transform = self.transform_embeddings[aug_name](aug_param)
            elif aug_name in [
                "brightness",
                "contrast",
                "saturation",
                "hue",
                "gamma",
                "hed",
            ]:
                z_transform = self.transform_embeddings[aug_name](
                    aug_param[..., None].float()
                )
            else:
                raise ValueError(
                    f"{aug_name} is not a valid augmentation parameter name"
                )
            # Add positional embedding if specified
            if self.use_transform_pos_embeddings:
                if self.positional_encoding_type == "learnable":
                    pos_index = torch.as_tensor(pos, device=aug_param.device)
                    pos_embedding = self.sequence_pos_embedding(pos_index)
                elif self.positional_encoding_type == "sinusoidal":
                    pos_embedding = self.sequence_pos_embedding[pos].to(
                        aug_param.device
                    )
                else:
                    raise ValueError(
                        f"Invalid positional_encoding_type: {self.positional_encoding_type}"
                    )
                z_transform_with_pos = z_transform + pos_embedding
                z_transforms.append(z_transform_with_pos)
            else:
                z_transforms.append(z_transform)

        # Stack the list of embeddings along a new dimension
        z_transforms = torch.stack(z_transforms, dim=1)
        return z_transforms

    def sample_aug_params(
        self,
        batch_size: int,
        device: torch.device = torch.device("cuda"),
        mode: Literal["instance_wise", "wsi_wise"] = "wsi_wise",
    ):
        """
        Sample random augmentation parameters and their relative positions.

        If a transform from the supported list is missing in self.aug_param_names,
        include it with zero values and append it at unique tail positions.
        """
        if mode not in ("instance_wise", "wsi_wise"):
            raise ValueError('mode must be "instance_wise" or "wsi_wise"')

        supported_aug_names = [
            "rotation",
            "crop",
            "h_flip",
            "v_flip",
            "gaussian_blur",
            "erosion",
            "dilation",
            "brightness",
            "contrast",
            "saturation",
            "hue",
            "gamma",
            "hed",
        ]

        canonical_names = sorted(self.transforms_parameters.keys())
        num_transforms = len(canonical_names)

        # Determine which supported transforms are missing from the current configuration.
        # For any missing transform, we will still include it in augmentation_parameters
        # so that the downstream model sees a consistent set of transforms.
        # These missing transforms are initialized with zero values (i.e., identity / no-op)
        # and assigned unique tail positions after all configured transforms.
        missing_names = [n for n in supported_aug_names if n not in canonical_names]
        required_positions = num_transforms + len(missing_names)

        # Build permutation/positions for configured transforms only
        if mode == "instance_wise":
            permutation_matrix = (
                torch.stack(
                    [
                        torch.randperm(num_transforms, device=device)
                        for _ in range(batch_size)
                    ],
                    dim=0,
                )
                if num_transforms > 0
                else torch.empty((batch_size, 0), dtype=torch.long, device=device)
            )
        else:  # wsi_wise
            if num_transforms > 0:
                single_permutation = torch.randperm(num_transforms, device=device)
                permutation_matrix = single_permutation.unsqueeze(0).repeat(
                    batch_size, 1
                )
            else:
                permutation_matrix = torch.empty(
                    (batch_size, 0), dtype=torch.long, device=device
                )

        positions_matrix = (
            torch.argsort(permutation_matrix, dim=1)
            if num_transforms > 0
            else torch.empty((batch_size, 0), dtype=torch.long, device=device)
        )

        augmentation_parameters = OrderedDict()
        # --- sample configured transforms as before ---
        for transform_index, name in enumerate(canonical_names):
            config = self.transforms_parameters[name]

            if name == "rotation":
                probability = float(config)
                if mode == "instance_wise":
                    apply_mask = torch.rand(batch_size, device=device) < probability
                    random_angles = torch.randint(0, 4, (batch_size,), device=device)
                    random_angles[~apply_mask] = 0
                    value_tensor = random_angles
                else:
                    apply = random.random() < probability
                    angle = random.randint(1, 3) if apply else 0
                    value_tensor = torch.full(
                        (batch_size,), angle, dtype=torch.int64, device=device
                    )

            elif name == "crop":
                probability = float(config)
                if mode == "instance_wise":
                    apply_mask = torch.rand(batch_size, device=device) < probability
                    random_crops = torch.randint(0, 5, (batch_size,), device=device)
                    random_crops[~apply_mask] = 0
                    value_tensor = random_crops
                else:
                    apply = random.random() < probability
                    crop_code = random.randint(1, 4) if apply else 0
                    value_tensor = torch.full(
                        (batch_size,), crop_code, dtype=torch.int64, device=device
                    )

            elif name in ("h_flip", "v_flip", "gaussian_blur", "erosion", "dilation"):
                probability = float(config)
                if mode == "instance_wise":
                    value_tensor = (
                        torch.rand(batch_size, device=device) < probability
                    ).int()
                else:
                    bit = int(random.random() < probability)
                    value_tensor = torch.full(
                        (batch_size,), bit, dtype=torch.int32, device=device
                    )

            elif name in (
                "brightness",
                "contrast",
                "saturation",
                "hue",
                "gamma",
                "hed",
            ):
                lower_bound, upper_bound = map(float, config)
                if mode == "instance_wise":
                    value_tensor = torch.empty(batch_size, device=device).uniform_(
                        lower_bound, upper_bound
                    )
                else:
                    scalar_value = random.uniform(lower_bound, upper_bound)
                    value_tensor = torch.full(
                        (batch_size,), scalar_value, dtype=torch.float32, device=device
                    )

            else:
                raise ValueError(f"'{name}' is not a recognised augmentation name")

            position_tensor = positions_matrix[:, transform_index]
            augmentation_parameters[name] = (value_tensor, position_tensor)

        for i, name in enumerate(missing_names):
            if name in ("rotation", "crop"):
                zeros = torch.zeros(batch_size, dtype=torch.int64, device=device)
            elif name in ("h_flip", "v_flip", "gaussian_blur", "erosion", "dilation"):
                zeros = torch.zeros(batch_size, dtype=torch.int32, device=device)
            else:  # continuous
                zeros = torch.zeros(batch_size, dtype=torch.float32, device=device)

            tail_pos = num_transforms + i  # unique: K, K+1, ..., K+M-1
            pos = torch.full((batch_size,), tail_pos, dtype=torch.long, device=device)
            augmentation_parameters[name] = (zeros, pos)

        return augmentation_parameters

    def forward(self, x, aug_params, **kwargs):
        """
        Forward pass: embed features, apply transformer blocks, and produce output.

        :param x: Input tensor of shape (B, input_dim).
        :param aug_params: Augmentation parameters from sample_aug_params.
        :return: Output tensor of shape (B, input_dim).
        """

        x = x[:, None, :]

        x = x.view(x.shape[0], self.chunk_size, self.embed_dim)
        pos_embeddings = self.chunk_pos_embeddings_buffer.unsqueeze(0)
        x = x + pos_embeddings
        z = self.forward_aug_params_embed(aug_params)

        for block in self.blocks:
            x = block(x, z)
        x = self.norm(x)

        x = x.view(x.shape[0], 1, -1)
        x = self.head(x)
        x = x[:, 0, :]
        return x
