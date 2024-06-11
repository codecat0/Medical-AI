#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :loss_utils.py
@Author :CodeCat
@Date   :2024/6/11 15:07
"""
import torch
import torch.nn.functional as F


def flatten(tensor):
    """
    将给定的tensor展平，使得通道轴位于最前面。

    Args:
        tensor (torch.Tensor): 需要展平的tensor，其shape为(N, C, D, H, W)。

    Returns:
        torch.Tensor: 展平后的tensor，其shape为(C, N * D * H * W)。

    """
    axis_order = (1, 0) + tuple(range(2, len(tensor.shape)))
    transposed = tensor.permute(axis_order).contiguous()
    return torch.flatten(transposed, start_dim=1, end_dim=-1)


def class_weights(tensor):
    """
    计算每个类别的权重，根据输入的tensor进行softmax计算后，对每个类别的权重进行归一化处理。

    Args:
        tensor (Tensor): 输入的tensor，shape为[N, C]，其中N为batch_size，C为类别数。

    Returns:
        Tensor: 每个类别的权重，shape为[C]。

    """
    tensor = F.softmax(tensor, dim=1)
    flattened = flatten(tensor)
    nominator = (1. - flattened).sum(-1)
    denominator = flattened.sum(-1)
    class_weight = denominator / nominator
    class_weight.requries_grad = False
    return class_weight
