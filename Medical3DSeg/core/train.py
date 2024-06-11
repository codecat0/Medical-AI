#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :train.py
@Author :CodeCat
@Date   :2024/6/11 16:16
"""
import os
import time
import shutil
from collections import deque
import torch
from torch.utils.data import DataLoader

from Medical3DSeg.utils import logger, TimeAverager, calculate_eta, resume


def check_logits_losses(logits_list, losses):
    len_logits = len(logits_list)
    len_losses = len(losses['types'])

    if len_logits != len_losses:
        raise ValueError(f'The number of logits {len_logits} is not equal to the number of losses {len_losses}')


def loss_computation(logits_list, labels, losses):
    check_logits_losses(logits_list, losses)
    loss_list = []
    for i in range(len(logits_list)):
        logits = logits_list[i]
        loss_i = losses['types'][i]
        loss_list.append(losses['coef'][i] * loss_i(logits, labels))
    return loss_list


def train(
        model,
        train_dataset,
        val_dataset=None,
        optimizer=None,
        save_dir='output',
        iters=10000,
        batch_size=2,
        resume_model=None,
        save_interval=1000,
        log_iters=10,
        num_workers=0,
        use_vdl=False,
        losses=None,
        keep_checkpoint_max=5,
        test_config=None,
        fp16=False,
):
    model.train()

    start_iter = 0
    if resume_model is not None:
        start_iter = resume(model, optimizer, resume_model)

    if not os.path.isdir(save_dir):
        if os.path.exists(save_dir):
            os.remove(save_dir)
        os.makedirs(save_dir)

    loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )

    if fp16:
        logger.info('Use amp to train')
        scaler = torch.cuda.amp.GradScaler(init_scale=1024)

    if use_vdl:
        from torch.utils.tensorboard import SummaryWriter
        log_writer = SummaryWriter(save_dir)

    avg_loss = 0.0
    avg_loss_list = []
    iters_per_epoch = len(loader)
    best_mean_iou = -1.0
    best_model_iter = -1
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    save_models = deque()
    batch_start = time.time()

    iter = start_iter
    while iter < iters:
        for data in loader:
            iter += 1
            if iter > iters:
                break
            reader_cost_averager.record(time.time() - batch_start)
            images = data[0]
            labels = data[1]

            if fp16:
                with torch.cuda.amp.autocast(enabled=True):
                    logits_list = model(images)
                    loss_list = loss_computation(logits_list, labels, losses)
                    loss = sum(loss_list)

                scaled = scaler.scale(loss)
                scaled.backward()
                scaler.step(optimizer)
            else:
                logits_list = model(images)
                loss_list = loss_computation(logits_list, labels, losses)
                loss = sum(loss_list)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # TODO: update lr
            lr = optimizer.param_groups[0]['lr']
            # lr_scheduler.step()

            avg_loss += loss.numpy()[0]
            if not avg_loss_list:
                avg_loss_list.append(l.numpy() for l in loss_list)
            batch_cost_averager.record(time.time() - batch_start, num_samples=batch_size)

            if iter % log_iters == 0:
                avg_loss /= log_iters
                avg_loss_list = [l[0] / log_iters for l in avg_loss_list]
                remain_iters = iters - iter
                avg_train_batch_cost = batch_cost_averager.get_average()
                avg_train_reader_cost = reader_cost_averager.get_average()
                eta = calculate_eta(remain_iters, avg_train_batch_cost)
                logger.info(
                    "[TRAIN] epoch {}, iter {}/{}, loss: {:.4f}, lr: {:.6f}, batch_cost: {:.4f}, reader_cost: {:.5f}" 
                    "ips: {:.4f}, samples/sec | ETA {}".format(
                        (iter - 1) // iters_per_epoch + 1, iter, iters,
                        avg_loss, lr, avg_train_batch_cost,
                        avg_train_reader_cost,
                        batch_cost_averager.get_ips_average(), eta
                    )
                )

                if use_vdl:
                    log_writer.add_scalar('Train/loss', avg_loss, iter)
                    if len(avg_loss_list) > 1:
                        avg_loss_dict = {}
                        for i, value in enumerate(avg_loss_list):
                            avg_loss_dict['loss_' + str(i)] = value
                        for key, value in avg_loss_dict.items():
                            log_tag = 'Train/' + key
                            log_writer.add_scalar(log_tag, value, iter)

                    log_writer.add_scalar('Train/lr', lr, iter)
                    log_writer.add_scalar('Train/batch_cost', avg_train_batch_cost, iter)
                    log_writer.add_scalar('Train/reader_cost', avg_train_reader_cost, iter)

                avg_loss = 0.0
                avg_loss_list = []
                reader_cost_averager.reset()
                batch_cost_averager.reset()

            if (iter % save_interval == 0 or iter == iters) and (val_dataset is not None):
                num_workers = 1 if num_workers > 0 else 0

                if test_config is None:
                    test_config = {}

                # TODO: evaluate on validation set

                model.train()

            if iter % save_interval == 0 or iter == iters:
                current_save_dir = os.path.join(save_dir, 'iter_{}'.format(iter))
                if not os.path.exists(current_save_dir):
                    os.makedirs(current_save_dir)
                torch.save(model.state_dict(), os.path.join(current_save_dir, 'model.pt'))
                torch.save(optimizer.state_dict(), os.path.join(current_save_dir, 'optimizer.pt'))
                save_models.append(current_save_dir)

                if len(save_models) > keep_checkpoint_max > 0:
                    model_to_remove = save_models.popleft()
                    shutil.rmtree(model_to_remove)

                if val_dataset is not None:
                    # TODO: validate the dataset
                    pass

            batch_start = time.time()

    time.sleep(0.5)
    if use_vdl:
        log_writer.close()
