import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Compact 2D U-Net. Input: four registered MRI modalities; output: tumor logit."""
    def __init__(self, in_channels=4, base_channels=16):
        super().__init__()
        widths = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.enc1 = ConvBlock(in_channels, widths[0])
        self.enc2 = ConvBlock(widths[0], widths[1])
        self.enc3 = ConvBlock(widths[1], widths[2])
        self.bottleneck = ConvBlock(widths[2], widths[3])
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(widths[3], widths[2], 2, stride=2)
        self.dec3 = ConvBlock(widths[2] * 2, widths[2])
        self.up2 = nn.ConvTranspose2d(widths[2], widths[1], 2, stride=2)
        self.dec2 = ConvBlock(widths[1] * 2, widths[1])
        self.up1 = nn.ConvTranspose2d(widths[1], widths[0], 2, stride=2)
        self.dec1 = ConvBlock(widths[0] * 2, widths[0])
        self.head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        c = self.enc3(self.pool(b))
        d = self.bottleneck(self.pool(c))
        x = self.dec3(torch.cat((self.up3(d), c), dim=1))
        x = self.dec2(torch.cat((self.up2(x), b), dim=1))
        x = self.dec1(torch.cat((self.up1(x), a), dim=1))
        return self.head(x)


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    score = (2 * (probs * targets).sum(dims) + eps) / (probs.sum(dims) + targets.sum(dims) + eps)
    return 1 - score.mean()
