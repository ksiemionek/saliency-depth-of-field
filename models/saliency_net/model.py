import torch
import torch.nn as nn
from transformers import AutoModel


class ConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.dropout1 = nn.Dropout2d(dropout)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_channels)

        self.act = nn.GELU()

        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_channels),
        )

    def forward(self, x):
        skip = self.skip(x)
        x = self.act(self.gn1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.gn2(self.conv2(x))
        x = x + skip
        x = self.act(x)
        return x


class SaliencyNet(nn.Module):
    def __init__(self, model_name, dropout):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        for param in self.encoder.parameters():
            param.requires_grad = False

        c_dim = self.encoder.config.hidden_size

        proj_dim = 64

        self.proj_f1 = nn.Sequential(
            nn.Conv2d(c_dim, proj_dim, 1), nn.GroupNorm(8, proj_dim), nn.GELU()
        )
        self.proj_f2 = nn.Sequential(
            nn.Conv2d(c_dim, proj_dim, 1), nn.GroupNorm(8, proj_dim), nn.GELU()
        )
        self.proj_f3 = nn.Sequential(
            nn.Conv2d(c_dim, proj_dim, 1), nn.GroupNorm(8, proj_dim), nn.GELU()
        )
        self.proj_f4 = nn.Sequential(
            nn.Conv2d(c_dim, proj_dim, 1), nn.GroupNorm(8, proj_dim), nn.GELU()
        )

        self.reduce = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )

        self.decoder_head = ConvResBlock(128, 128, dropout=dropout)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def reshape(self, tokens, h, w):
        seq_len = h * w
        tokens = tokens[:, -seq_len:, :]
        B, _, C = tokens.shape
        return tokens.transpose(1, 2).reshape(B, C, h, w)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x):
        patch_h = x.shape[2] // 16
        patch_w = x.shape[3] // 16

        with torch.no_grad():
            outputs = self.encoder(
                pixel_values=x,
                output_hidden_states=True,
                interpolate_pos_encoding=True,
            )
            hidden_states = outputs.hidden_states

        f1 = self.reshape(hidden_states[3], patch_h, patch_w)
        f2 = self.reshape(hidden_states[6], patch_h, patch_w)
        f3 = self.reshape(hidden_states[9], patch_h, patch_w)
        f4 = self.reshape(hidden_states[12], patch_h, patch_w)

        f1 = self.proj_f1(f1)
        f2 = self.proj_f2(f2)
        f3 = self.proj_f3(f3)
        f4 = self.proj_f4(f4)

        out = torch.cat([f1, f2, f3, f4], dim=1)
        out = self.reduce(out)

        out = self.decoder_head(out)
        return self.upsample(out)
