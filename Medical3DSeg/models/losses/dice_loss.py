#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :dice_loss.py
@Author :CodeCat
@Date   :2024/6/11 15:32
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from Medical3DSeg.models.losses import flatten


class DiceLoss(nn.Module):
    def __init__(self, sigmoid_norm=True, weight=None):
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.EPS = 1e-5
        if sigmoid_norm:
            self.norm = nn.Sigmoid()
        else:
            self.norm = nn.Softmax(dim=1)

    @staticmethod
    def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
        assert input.shape == target.shape, "'input' and 'target' must have the same shape but"\
            "input is {} and target is {}".format(input.shape, target.shape)

        input = flatten(input)  # C, N*D*H*W
        target = flatten(target)  # C, N*D*H*W
        target = target.float()

        intersect = (input * target).sum(dim=-1)
        if weight is not None:
            intersect = weight * intersect

        denominator = (input * input).sum(dim=-1) + (target * target).sum(dim=-1)

        return 2 * (intersect / denominator.clamp(min=epsilon))

    def forward(self, logits, labels):
        """
        计算Dice Loss并返回每个通道的Dice系数。

        Args:
            logits (torch.Tensor): 网络输出的logits张量，shape为(N, C, D, H, W)。
            labels (torch.Tensor): 真实标签张量，shape为(N, D, H, W)或(N)，标签值应为整数类型。

        Returns:
            Tuple[torch.Tensor, np.ndarray]: 一个包含两个元素的元组，分别为Dice Loss和每个通道的Dice系数。
                - Dice Loss (torch.Tensor): Dice Loss值，shape为(1)。
                - per_channel_dice (np.ndarray): 每个通道的Dice系数，shape为(C)。

        """
        labels = labels.long()

        if len(logits.shape) == 4:
            logits = logits.unsqueeze(0)

        labels_one_hot = F.one_hot(labels, num_classes=logits.shape[1])
        labels_one_hot = labels_one_hot.permute(0, 4, 1, 2, 3).contiguous().float()

        logits = self.norm(logits)

        per_channel_dice = self.compute_per_channel_dice(logits, labels_one_hot, weight=self.weight)

        dice_loss = (1. - torch.mean(per_channel_dice))
        per_channel_dice = per_channel_dice.detach().cpu().numpy()

        return dice_loss, per_channel_dice


if __name__ == '__main__':
    inputs = torch.randn(1, 5, 32, 64, 64)
    targets = torch.randint(0, 5, (1, 32, 64, 64))
    loss_fn = DiceLoss()
    print(loss_fn(inputs, targets))
