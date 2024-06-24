#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :infer.py
@Author :CodeCat
@Date   :2024/6/12 15:04
"""
import collections.abc
from itertools import combinations

import numpy as np
import cv2
import torch
import torch.nn.functional as F


def get_reverse_list(ori_shape, transforms):
    reverse_list = []
    d, h, w = ori_shape[0], ori_shape[1], ori_shape[2]
    for op in transforms:
        if op.__class__.__name__ in ['Resize3D']:
            reverse_list.append(('resize', (d, h, w)))
            d, h, w = op.size[0], op.size[1], op.size[2]

    return reverse_list


def reverse_transform(pred, ori_shape, transforms, mode='trilinear'):
    """
    对预测结果进行反变换，得到原始图像的预测结果
    Args:
        pred (torch.Tensor): 预测结果，形状为[1, num_classes, D, H, W]
        ori_shape (tuple): 原图的大小，形如(D, H, W)
        transforms (list): 预处理流水线中的变换操作列表
        mode (str): 插值模式，默认为'trilinear'
    Returns:
        torch.Tensor: 原始图像的预测结果，形状为[1, num_classes, D, H, W]
    """
    reverse_list = get_reverse_list(ori_shape, transforms)
    for item in reverse_list[::-1]:
        if item[0] == 'resize':
            d, h, w = item[1][0], item[1][1], item[1][2]
            pred = F.interpolate(
                pred, size=(d, h, w), mode=mode, align_corners=True)
        else:
            raise Exception("UNexpected info '{}' in im_info.".format(item[0]))
    return pred


def inference(
        model,
        image,
        ori_shape=None,
        transforms=None,
        sw_num=None
):
    if sw_num:
        pass
    else:
        logits = model(image)

    if ori_shape is not None and ori_shape != logits.shape[2:]:
        logits = reverse_transform(
            logits, ori_shape=ori_shape, transforms=transforms)

    pred = torch.argmax(logits, dim=1, keepdim=True)
    return pred, logits


