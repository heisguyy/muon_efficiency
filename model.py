"""
This script was adapted from https://github.com/heisguyy/vit/blob/main/model.py
"""

import math
from dataclasses import dataclass

# pylint: disable=not-callable
from typing import Optional, Tuple

import einops
import torch
from torch import nn


@dataclass
class ViTConfig:  # pylint: disable=too-many-instance-attributes
    """All the configuration parameters for the ViT model"""

    attention_probs_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.0
    hidden_size: int = 768
    image_size: int = 224
    initializer_range: float = 0.02
    intermediate_size: int = 3072
    layer_norm_eps: float = 1e-12
    num_attention_heads: int = 12
    num_channels: int = 3
    num_hidden_layers: int = 12
    patch_size: int = 16
    num_labels: int = 1000


class ViTPatchEmbeddings(nn.Module):
    """Patch embeddings for the Vision Transformer model"""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.embedding_dimension = config.hidden_size

        self.projection = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embedding_dimension,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        patched_image = self.projection(x)
        # flatten and transpose: channels is the embedding dimension and
        # height * width will give the number of patches
        output_projection = einops.rearrange(
            patched_image,
            "batch channels height width -> batch (height width) channels",
        ).contiguous()
        return output_projection


class ViTEmbeddings(nn.Module):
    """Embeddings for the ViT model"""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.patch_embeddings = ViTPatchEmbeddings(config)
        self.embedding_dimension = config.hidden_size
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_size)
        )  # global token

        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.num_patches + 1, self.embedding_dimension)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        _, _, height, width = x.shape
        patch_embeddings = self.patch_embeddings(x)
        global_token = self.cls_token.expand(patch_embeddings.shape[0], -1, -1)
        # add the global token
        patch_embeddings = torch.cat((global_token, patch_embeddings), dim=1)
        if height > self.config.image_size or width > self.config.image_size:
            position_embeddings = self.__interpolate_positional_embeddings(
                height, width
            )
            embeddings = patch_embeddings + position_embeddings
        else:
            embeddings = patch_embeddings + self.position_embeddings
        return embeddings

    def __interpolate_positional_embeddings(
        self,
        new_height: int,
        new_width: int,
    ) -> None:
        """Interpolate positional embeddings to accommodate larger images"""
        global_token_position_embedding = self.position_embeddings[:, 0, :]
        old_position_embeddings = self.position_embeddings[:, 1:, :]
        old_position_embeddings = einops.rearrange(
            old_position_embeddings,
            "batch (height width) channels -> batch channels height width",
            height=int(math.sqrt(old_position_embeddings.shape[1])),
            width=int(math.sqrt(old_position_embeddings.shape[1])),
        ).contiguous()
        new_height //= self.config.patch_size
        new_width //= self.config.patch_size
        new_position_embeddings = nn.functional.interpolate(
            old_position_embeddings,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )
        new_position_embeddings = einops.rearrange(
            new_position_embeddings,
            "batch channels height width -> batch (height width) channels",
        ).contiguous()
        new_position_embeddings = torch.cat(
            [
                global_token_position_embedding.unsqueeze(0),
                new_position_embeddings
            ],
            dim=1
        )
        return new_position_embeddings


class ViTSelfOutput(nn.Module):
    """Output layer for the self-attention layer in the ViT model"""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        hidden_state = self.dense(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return hidden_state


class ViTSdpaSelfAttention(
    nn.Module
):  # pylint: disable=too-many-instance-attributes
    """Scaled Dot-Product Attention for the ViT model"""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads

        self.query = nn.Linear(self.embed_dim, self.embed_dim)
        self.key = nn.Linear(self.embed_dim, self.embed_dim)
        self.value = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(
        self, hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass"""
        attention_query = self.query(hidden_state)
        attention_key = self.key(hidden_state)
        attention_value = self.value(hidden_state)
        # rearrange the query, key and value tensors to make it multihead
        # attention
        multihead_query = einops.rearrange(
            attention_query,
            "batch patches (heads head_dim) -> batch heads patches head_dim",
            heads=self.num_attention_heads,
            head_dim=self.head_dim,
        ).contiguous()
        multihead_key = einops.rearrange(
            attention_key,
            "batch patches (heads head_dim) -> batch heads head_dim patches",
            heads=self.num_attention_heads,
            head_dim=self.head_dim,
        ).contiguous()
        multihead_value = einops.rearrange(
            attention_value,
            "batch patches (heads head_dim) -> batch heads patches head_dim",
            heads=self.num_attention_heads,
            head_dim=self.head_dim,
        ).contiguous()
        attention = torch.matmul(multihead_query, multihead_key) * (
            self.head_dim**-0.5
        )

        attention = nn.functional.softmax(
            attention, dim=-1, dtype=torch.float32
        ).to(attention.dtype)
        attention = self.dropout(attention)
        attention_output = torch.matmul(attention, multihead_value)
        attention_output = einops.rearrange(
            attention_output,
            "batch heads patches head_dim -> batch patches (heads head_dim)",
        ).contiguous()

        return attention_output


class ViTSdpaAttention(nn.Module):
    """Attention layer for the ViT model"""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.attention = ViTSdpaSelfAttention(config)
        self.output = ViTSelfOutput(config)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        attention_output = self.attention(hidden_state)
        attention_output = self.output(attention_output)
        return attention_output


class GELUActivation(nn.Module):
    """GELU activation function"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return nn.functional.gelu(x)


class ViTIntermediate(nn.Module):
    """MLP for the ViT model"""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = GELUActivation()

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        hidden_state = self.dense(hidden_state)
        hidden_state = self.intermediate_act_fn(hidden_state)
        return hidden_state


class ViTOutput(nn.Module):
    """MLP for the ViT model"""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        hidden_state = self.dense(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return hidden_state


class ViTLayer(nn.Module):
    """Encoder layer for the ViT model"""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.attention = ViTSdpaAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(
            self.embed_dim, eps=config.layer_norm_eps
        )
        self.layernorm_after = nn.LayerNorm(
            self.embed_dim, eps=config.layer_norm_eps
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        residual_1 = hidden_state
        last_hidden_state = self.layernorm_before(hidden_state)
        last_hidden_state = self.attention(last_hidden_state)
        last_hidden_state = last_hidden_state + residual_1
        residual_2 = last_hidden_state
        last_hidden_state = self.layernorm_after(last_hidden_state)
        last_hidden_state = self.intermediate(last_hidden_state)
        last_hidden_state = self.output(last_hidden_state)
        last_hidden_state = last_hidden_state + residual_2
        return last_hidden_state


class ViTEncoder(nn.Module):
    """Encoder for the ViT model"""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.layer = nn.ModuleList(
            [ViTLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        hidden_state = inputs_embeds
        for layer in self.layer:
            hidden_state = layer(hidden_state)
        return hidden_state


class ViTModel(nn.Module):
    """ViT core model"""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.embeddings = ViTEmbeddings(config)
        self.encoder = ViTEncoder(config)
        self.layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        hidden_state = self.embeddings(x)
        hidden_state = self.encoder(hidden_state)
        hidden_state = self.layernorm(hidden_state[:, 0, :])
        return hidden_state


class ViTForImageClassification(nn.Module):
    """ViT model for image classification"""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.vit = ViTModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        hidden_state = self.vit(x)
        logits = self.classifier(hidden_state)
        return logits
