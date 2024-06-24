#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :loss_utils.py
@Author :CodeCat
@Date   :2024/6/12 15:42
"""


def loss_computation(logits, labels, losses):
    loss_list = []
    per_channel_dice = None
    if losses.__class__.__name__ == 'MixedLoss':
        mixed_loss_list, per_channel_dice = losses(logits, labels)
        for mixed_loss in mixed_loss_list:
            loss_list.append(mixed_loss)
    elif losses.__class__.__name__ == 'DiceLoss':
        dice_loss, per_channel_dice = losses(logits, labels)
        loss_list.append(dice_loss)
    else:
        loss = losses(logits, labels)
        loss_list.append(loss)
    return loss_list, per_channel_dice
