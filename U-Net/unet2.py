import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()

        # Encoder
        self.enc1 = self.conv_block(input_channels, 16)  # (B, 16, H, W)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = self.conv_block(16, 32)              # (B, 32, H/2, W/2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = self.conv_block(32, 64)              # (B, 64, H/4, W/4)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self.conv_block(64, 128)

        # Decoder with skip connections
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(64, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(32, 16)

        # Output
        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)  # Binary mask output

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)     # (B, 16, H, W)
        e2 = self.enc2(self.pool1(e1))  # (B, 32, H/2, W/2)
        e3 = self.enc3(self.pool2(e2))  # (B, 64, H/4, W/4)

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))  # (B, 128, H/8, W/8)

        # Decoder with skip connections
        d3 = self.up3(b)                     # (B, 64, H/4, W/4)
        d3 = torch.cat([d3, e3], dim=1)      # (B, 128, H/4, W/4)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)                    # (B, 32, H/2, W/2)
        d2 = torch.cat([d2, e2], dim=1)      # (B, 64, H/2, W/2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)                    # (B, 16, H, W)
        d1 = torch.cat([d1, e1], dim=1)      # (B, 32, H, W)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)              # (B, 1, H, W)
        return out  # → Sigmoid는 후처리 시 적용 (예: torch.sigmoid)

