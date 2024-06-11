#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :cross_entropy_loss.py
@Author :CodeCat
@Date   :2024/6/11 15:00
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from Medical3DSeg.models.losses import class_weights


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=255):
        super(CrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.EPS = 1e-8
        if weight is not None:
            self.weight = torch.FloatTensor(weight)
        else:
            self.weight = None

    def forward(self, logit, label):

        if len(logit.shape) == 4:
            logit = logit.unsqueeze(0)

        if self.weight is None:
            self.weight = class_weights(logit)

        if self.weight is not None and logit.shape[1] != len(self.weight):
            raise ValueError(
                'The number of weights = {} must be equal to the number of classes = {}'
                .format(len(self.weight), logit.shape[1])
            )

        loss = F.cross_entropy(
            logit + self.EPS,
            label.long(),
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='mean'
        )

        return loss


if __name__ == '__main__':
    inputs = torch.randn(1, 5, 32, 64, 64)
    targets = torch.randint(0, 5, (1, 32, 64, 64))
    loss_fn = CrossEntropyLoss()
    print(loss_fn(inputs, targets))
