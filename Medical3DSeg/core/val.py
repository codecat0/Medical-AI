#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :val.py
@Author :CodeCat
@Date   :2024/6/12 0:19
"""
import os
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Medical3DSeg.utils import TimeAverager, calculate_eta, logger, metrics, Progbar, loss_computation
from Medical3DSeg.core import infer


def evaluate(
        model,
        eval_dataset,
        losses,
        num_workers=0,
        print_details=True,
        sw_num=None
):
    model.eval()
    loader = DataLoader(
        dataset=eval_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True
    )

    total_iters = len(loader)
    intersect_area_all = 0
    pred_area_all = 0
    label_area_all = 0

    if print_details:
        logger.info(
            "Start evaluating (total_samples: {}, total_iters: {})"
            .format(len(eval_dataset), total_iters)
        )

    progbar_val = Progbar(target=total_iters, verbose=1)
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()

    mdice = 0.0
    channel_dice_array = np.array([])
    loss_all = 0.0

    with torch.no_grad():
        for iter, (image, label) in enumerate(loader):
            reader_cost_averager.record(time.time() - batch_start)
            label = label.long()

            ori_shape = label.shape[-3:]

            if sw_num:
                pred, logits = infer.inference(
                    model,
                    image,
                    ori_shape=ori_shape,
                    transforms=eval_dataset.transforms.transforms,
                    sw_num=sw_num
                )
            else:
                pred, logits = infer.inference(
                    model,
                    image,
                    ori_shape=ori_shape,
                    transforms=eval_dataset.transforms.transforms
                )

            intersect_area, pred_area, label_area = metrics.calculate_area(
                pred,
                label,
                num_classes=eval_dataset.num_classes,
                ignore_index=eval_dataset.ignore_index
            )
            intersect_area_all += intersect_area
            pred_area_all += pred_area
            label_area_all += label_area

            loss, per_channel_dice = loss_computation(logits, label, losses)
            loss = sum(loss)

            loss_all += loss.numpy()
            mdice += np.mean(per_channel_dice)
            if channel_dice_array.size == 0:
                channel_dice_array = per_channel_dice
            else:
                channel_dice_array += per_channel_dice

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=len(label)
            )
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()

            if print_details:
                progbar_val.update(
                    iter+1,
                    [('batch_cost', batch_cost), ('reader_cost', reader_cost)]
                )
                reader_cost_averager.reset()
                batch_cost_averager.reset()

    mdice /= total_iters
    channel_dice_array /= total_iters
    loss_all /= total_iters

    class_iou, miou = metrics.mean_iou(
        intersect_area_all, pred_area_all, label_area_all
    )
    class_acc, acc = metrics.accuracy(
        intersect_area_all, pred_area_all
    )
    kappa = metrics.kappa(
        intersect_area_all, pred_area_all, label_area_all
    )

    result_dict = {
        'mdice': mdice,
        'miou': miou,
        'acc': acc,
        'kappa': kappa
    }

    if print_details:
        infor = "[EVAL] #Image: {}, Dice: {:.4f}, mIoU: {:.4f}, Acc: {:.4f}, Kappa: {:.4f}, Loss: {:6f}".format(
            len(eval_dataset), mdice, miou, acc, kappa, loss_all
        )
        logger.info(infor)
        logger.info("[EVAL] Class Dice: \n" + str(np.round(channel_dice_array, 4)))
        logger.info("[EVAL] Class IoU: \n" + str(np.round(class_iou, 4)))
        logger.info("[EVAL] Class Acc: \n" + str(np.round(class_acc, 4)))

    return result_dict