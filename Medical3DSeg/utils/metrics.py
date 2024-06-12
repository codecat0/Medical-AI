#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :metrics.py
@Author :CodeCat
@Date   :2024/6/12 0:38
"""
import numpy as np
import torch
import torch.nn.functional as F


def calculate_area(pred, label, num_classes, ignore_index=255):
    if len(pred.shape) == 5:
        pred = torch.squeeze(pred, dim=1)
    if len(label.shape) == 5:
        label = torch.squeeze(label, dim=1)
    if not pred.shape == label.shape:
        raise ValueError("pred and label should have the same shape, but got {} and {}".format(pred.shape, label.shape))

    mask = label != ignore_index
    pred = pred * mask
    label = label * mask
    pred = F.one_hot(pred.long(), num_classes=num_classes)
    label = F.one_hot(label.long(), num_classes=num_classes)

    pred_area = []
    label_area = []
    intersect_area = []

    for i in range(num_classes):
        pred_i = pred[..., i]
        label_i = label[..., i]
        pred_area_i = torch.sum(pred_i)
        label_area_i = torch.sum(label_i)
        intersect_area_i = torch.sum(pred_i * label_i)
        pred_area.append(pred_area_i)
        label_area.append(label_area_i)
        intersect_area.append(intersect_area_i)

    pred_area = torch.cat(pred_area, dim=0)
    label_area = torch.cat(label_area, dim=0)
    intersect_area = torch.cat(intersect_area, dim=0)
    return intersect_area, pred_area, label_area


def main_iou(intersect_area, pred_area, label_area):
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    laebl_area = label_area.numpy()

    union = pred_area + laebl_area - intersect_area
    class_iou = []

    for i in range(len(intersect_area)):
        if union[i] == 0:
            iou = 0
        else:
            iou = intersect_area[i] / union[i]
        class_iou.append(iou)

    miou = np.mean(class_iou)
    return np.array(class_iou), miou


def accuracy(intersect_area, pred_area):
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    class_acc = []
    for i in range(len(intersect_area)):
        if pred_area[i] == 0:
            acc = 0
        else:
            acc = intersect_area[i] / pred_area[i]
        class_acc.append(acc)
    maacc = np.sum(intersect_area) / np.sum(pred_area)
    return np.array(class_acc), maacc


def kappa(intersect_area, pred_area, label_area):
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    label_area = label_area.numpy()
    total_area = np.sum(label_area)
    po = np.sum(intersect_area) / total_area
    pe = np.sum(pred_area * label_area) / (total_area * total_area)
    kappa_coef = (po - pe) / (1 - pe)
    return kappa_coef
