import torch
import torch.nn as nn
from transformers import AutoModel

from models.saliency_net.model_config import FEATURE_LAYERS


class ConvResBlock(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, channels)
        self.drop = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.act(self.gn1(self.conv1(x)))
        x = self.drop(x)
        x = self.gn2(self.conv2(x))
        return self.act(x + res)


def projection_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=1),
        nn.GroupNorm(8, out_ch),
        nn.GELU(),
    )


def upsample_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.GELU(),
    )


class SaliencyNet(nn.Module):
    def __init__(self, model_name: str, dropout: float, decoder_dim: int):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        for param in self.encoder.parameters():
            param.requires_grad = False

        cfg = self.encoder.config
        self.patch_size = cfg.patch_size
        self.num_register_tokens = cfg.num_register_tokens
        c_dim = cfg.hidden_size

        self.proj = nn.ModuleList(
            [projection_block(c_dim, decoder_dim) for _ in FEATURE_LAYERS]
        )

        self.decoder = ConvResBlock(decoder_dim, dropout=dropout)

        upsample_channels = [decoder_dim, 64, 32, 16]
        self.upsample = nn.Sequential(
            *[
                upsample_block(upsample_channels[i], upsample_channels[i + 1])
                for i in range(len(upsample_channels) - 1)
            ],
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def reshape(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        tokens = tokens[:, 1 + self.num_register_tokens :, :]
        B, _, C = tokens.shape
        return tokens.transpose(1, 2).reshape(B, C, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_h = x.shape[2] // self.patch_size
        patch_w = x.shape[3] // self.patch_size

        hidden_states = self.encoder(
            pixel_values=x,
            output_hidden_states=True,
            interpolate_pos_encoding=True,
        ).hidden_states

        out = sum(
            proj(self.reshape(hidden_states[i], patch_h, patch_w))
            for proj, i in zip(self.proj, FEATURE_LAYERS)
        )

        return self.upsample(self.decoder(out))
