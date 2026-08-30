import torch
import torch.nn as nn


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.conv(x)


class UNetChangeDetector(nn.Module):

    def __init__(self):

        super().__init__()

        # Encoder
        self.encoder1 = DoubleConv(6, 64)

        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = DoubleConv(64, 128)

        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)

        # Decoder
        self.upconv2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.decoder2 = DoubleConv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.decoder1 = DoubleConv(128, 64)

        # Output
        self.final_conv = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, image1, image2):

        # Concatenate images
        x = torch.cat(
            [image1, image2],
            dim=1
        )

        # Encoder
        e1 = self.encoder1(x)

        p1 = self.pool1(e1)

        e2 = self.encoder2(p1)

        p2 = self.pool2(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder
        u2 = self.upconv2(b)

        u2 = torch.cat([u2, e2], dim=1)

        d2 = self.decoder2(u2)

        u1 = self.upconv1(d2)

        u1 = torch.cat([u1, e1], dim=1)

        d1 = self.decoder1(u1)

        # Output
        output = self.final_conv(d1)

        output = self.sigmoid(output)

        return output