#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :unet3d.py
@Author :CodeCat
@Date   :2024/6/11 14:16
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        insert_channels = out_channels if in_channels > out_channels else out_channels // 2

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, insert_channels, 3, stride=1, padding=1),
            nn.BatchNorm3d(insert_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(insert_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.downsample = nn.MaxPool3d(2, stride=2)

    def forward(self, x):
        x = self.conv_block(x)
        out = self.downsample(x)
        return x, out


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, sample=True):
        super(Up, self).__init__()
        if sample:
            self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.upsample = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)

        self.conv_block = ConvBlock(in_channels + in_channels // 2, out_channels)

    def forward(self, x, conv):
        x = self.upsample(x)
        x = torch.cat([x, conv], dim=1)
        x = self.conv_block(x)
        return x


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, num_filters=32, num_classes=5, sample=True, has_dropout=False):
        super(UNet3D, self).__init__()

        self.down_1 = Down(in_channels, num_filters)
        self.down_2 = Down(num_filters, num_filters * 2)
        self.down_3 = Down(num_filters * 2, num_filters * 4)
        self.down_4 = Down(num_filters * 4, num_filters * 8)

        self.bridge = ConvBlock(num_filters * 8, num_filters * 16)

        self.up_1 = Up(num_filters * 16, num_filters * 8, sample)
        self.up_2 = Up(num_filters * 8, num_filters * 4, sample)
        self.up_3 = Up(num_filters * 4, num_filters * 2, sample)
        self.up_4 = Up(num_filters * 2, num_filters, sample)

        self.conv_class = nn.Conv3d(num_filters, num_classes, kernel_size=1)

        self.has_dropout = has_dropout
        self.droupout = nn.Dropout3d(0.5)

    def forward(self, x):
        conv1, x = self.down_1(x)
        conv2, x = self.down_2(x)
        conv3, x = self.down_3(x)
        conv4, x = self.down_4(x)

        x = self.bridge(x)
        if self.has_dropout:
            x = self.droupout(x)

        x = self.up_1(x, conv4)
        x = self.up_2(x, conv3)
        x = self.up_3(x, conv2)
        x = self.up_4(x, conv1)
        if self.has_dropout:
            x = self.droupout(x)

        out = self.conv_class(x)
        return out


if __name__ == '__main__':
    inputs = torch.randn(1, 1, 32, 512, 512)
    model = UNet3D(in_channels=1, num_filters=32, num_classes=5)
    outputs = model(inputs)
    print(outputs.shape)


