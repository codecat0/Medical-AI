#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :mixed_loss.py
@Author :CodeCat
@Date   :2024/6/11 15:57
"""
import torch.nn as nn


class MixedLoss(nn.Module):
    def __init__(self, losses, coef):
        super(MixedLoss, self).__init__()
        if not isinstance(losses, list):
            raise ValueError("The losses should be a list.")
        if not isinstance(coef, list):
            raise ValueError("The coef should be a list.")
        len_losses = len(losses)
        len_coef = len(coef)
        if len_losses != len_coef:
            raise ValueError("The length of losses and coef should be the same.")

        self.losses = losses
        self.coef = coef

    def forward(self, logits, labels):
        loss_list = []
        per_channel_dice = None
        for i, loss in enumerate(self.losses):
            output = loss(logits, labels)
            if type(loss).__name__ == "DiceLoss":
                output, per_channel_dice = output
            loss_list.append(output * self.coef[i])
        return loss_list, per_channel_dice