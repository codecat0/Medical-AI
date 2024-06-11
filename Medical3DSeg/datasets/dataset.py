#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :dataset.py
@Author :CodeCat
@Date   :2024/6/11 9:46
"""
import os
import numpy as np
import SimpleITK as sitk

from torch.utils.data import Dataset
from Medical3DSeg.transforms.transform import Compose


class MedicalDataset(Dataset):
    def __init__(self,
                 dataset_root,
                 result_dir,
                 transforms,
                 num_classes,
                 mode='train',
                 ignore_index=255,
                 dataset_json_path='',
                 repeat_times=10):
        super(MedicalDataset, self).__init__()
        self.dataset_root = dataset_root
        self.result_dir = result_dir
        self.transforms = Compose(transforms)
        self.file_list = list()
        self.mode = mode.lower()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.dataset_json_path = dataset_json_path

        if not os.path.exists(self.dataset_root):
            raise ValueError(f'{self.dataset_root} is not exist!')

        if mode == 'train':
            file_path = os.path.join(self.dataset_root, 'train_list.txt')
        elif mode == 'val':
            file_path = os.path.join(self.dataset_root, 'val_list.txt')
        elif mode == 'test':
            file_path = os.path.join(self.dataset_root, 'test_list.txt')
        else:
            raise ValueError(
                "'mode' should be 'train', 'val' or 'test', but got {}".format(mode)
            )

        with open(file_path, 'r') as f:
            for line in f.readlines():
                items = line.strip().split()
                if len(items) != 2:
                    raise Exception("File list format incorrect! It should be"
                                    " image_name lable_name\\n")
                else:
                    image_path = os.path.join(self.dataset_root, items[0])
                    label_path = os.path.join(self.dataset_root, items[1])
                self.file_list.append([image_path, label_path])

        if mode == 'train':
            self.file_list = self.file_list * repeat_times

    def __getitem__(self, idx):
        """
        根据索引获取图像和标签。

        Args:
            idx (int): 索引值，表示要获取的图像和标签的索引位置。

        Returns:
            tuple: 返回一个元组，包含两个元素，分别表示获取到的图像和标签。
                - 图像：经过转换的numpy数组形式的图像数据。
                - 标签：经过转换的numpy数组形式的标签数据。如果模式为测试模式，则仅返回图像和图像路径。

        """
        image_path, label_path = self.file_list[idx]
        img = sitk.ReadImage(image_path)
        label = sitk.ReadImage(label_path)
        img = np.array(sitk.GetArrayFromImage(img))
        label = np.array(sitk.GetArrayFromImage(label))
        if self.mode == 'test':
            img, _ = self.transforms(img=img)
            return img, image_path
        elif self.mode == 'val':
            img, label = self.transforms(img=img, label=label)
            return img, label
        else:
            img, label = self.transforms(img=img, label=label)
            return img, label

    def __len__(self):
        return len(self.file_list)
