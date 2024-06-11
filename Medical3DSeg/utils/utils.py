#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :utils.py
@Author :CodeCat
@Date   :2024/6/11 16:50
"""
import os
import torch
from Medical3DSeg.utils import logger


def resume(model, optimizer, resume_model):
    if resume_model is not None:
        logger.info('Resume model from {}'.format(resume_model))
        if os.path.exists(resume_model):
            resume_model = os.path.normpath(resume_model)
            model_path = os.path.join(resume_model, 'model.pt')
            para_satate_dict = torch.load(model_path)
            model.load_state_dict(para_satate_dict)
            optimizer_path = os.path.join(resume_model, 'optimizer.pt')
            optimizer_state_dict = torch.load(optimizer_path)
            optimizer.load_state_dict(optimizer_state_dict)

            last_iter = resume_model.split('_')[-1]
            last_iter = int(last_iter)
            return last_iter
        else:
            raise FileNotFoundError('Resume model not found in {}'.format(resume_model))
    else:
        logger.info('No model to resume')

