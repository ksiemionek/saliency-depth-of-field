import torch
import torch.nn as nn
import timm


class ConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.25):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(dropout)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = nn.GELU()

        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        skip = self.skip(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.bn2(self.conv2(x))
        x = x + skip
        x = self.act(x)
        return x


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels, dropout):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)

        self.conv2 = nn.Conv2d(channels, channels * 4, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(channels * 4)

        self.conv3 = nn.Conv2d(channels * 4, channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(channels)

        self.act = nn.GELU()

    def forward(self, x):
        skip = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.act(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x + skip
        x = self.act(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels_dec, in_channels_enc, out_channels, dropout):
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels_dec, in_channels_dec, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels_dec),
            nn.GELU(),
        )

        self.conv = ConvResBlock(
            in_channels_dec + in_channels_enc, out_channels, dropout
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SaliencyNet(nn.Module):
    def __init__(self, model_name, dropout):
        super().__init__()

        self.encoder = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )

        channels = self.encoder.feature_info.channels()
        c1, c2, c3, c4 = channels[0], channels[1], channels[2], channels[3]

        self.context = nn.Sequential(
            ConvNeXtBlock(c4, dropout=dropout),
            ConvNeXtBlock(c4, dropout=dropout),
        )

        self.up1 = UpBlock(c4, c3, 256, dropout)
        self.up2 = UpBlock(256, c2, 128, dropout)
        self.up3 = UpBlock(128, c1, 64, dropout)

        self.upsample = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.encoder(x)
        f1, f2, f3, f4 = features[0], features[1], features[2], features[3]

        x = self.context(f4)

        x = self.up1(x, f3)
        x = self.up2(x, f2)
        x = self.up3(x, f1)

        return self.upsample(x)
