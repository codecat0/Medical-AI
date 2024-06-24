#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :dataset.py
@Author :CodeCat
@Date   :2024/6/11 9:46
"""
import os
import SimpleITK as sitk

from torch.utils.data import Dataset
from Medical3DSeg.transforms.transform import Compose

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class MedicalDataset(Dataset):
    """
    初始化 MedicalDataset 类。

    Args:
        dataset_root (str): 数据集的根目录。
        transforms (list): 数据预处理流程。
        num_classes (int): 类别数。
        mode (str, optional): 数据集模式，可选值为 'train', 'val', 'test'，默认为 'train'。
        ignore_index (int, optional): 忽略的索引，默认为 255。
        dataset_json_path (str, optional): 数据集的 json 路径，默认为空字符串。
        repeat_times (int, optional): 训练集重复次数，默认为 10。
        win_center (int, optional): 窗口中心，默认为 40。
        win_width (int, optional): 窗口宽度，默认为 100。
        depth (int, optional): 数据集的深度，默认为 16。

    Returns:
        None

    Raises:
        ValueError: 如果数据集根目录不存在或数据集模式不在 ['train', 'val', 'test'] 中。
        Exception: 如果文件列表格式不正确。
    """

    def __init__(self,
                 dataset_root,
                 transforms,
                 num_classes,
                 mode='train',
                 ignore_index=255,
                 dataset_json_path='',
                 repeat_times=1,
                 win_center=40,
                 win_width=100,
                 depth=16):
        super(MedicalDataset, self).__init__()
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.file_list = list()
        self.mode = mode.lower()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.dataset_json_path = dataset_json_path
        self.win_center = win_center
        self.win_width = win_width
        self.depth = depth

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
                items = line.strip().split('\t')
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
        img = self.sikt_read_raw(image_path)
        label = self.sikt_read_raw(label_path)
        if self.mode == 'test':
            img, _ = self.transforms(img=img)
            return img, image_path
        elif self.mode == 'val':
            img, label = self.transforms(img=img, label=label)
            return img, label
        else:
            img, label = self.transforms(img=img, label=label)
            return img, label

    def sikt_read_raw(self, img_path):
        """
        读取医学图像，并返回其数组表示。

        Args:
            img_path (str): 医学图像的路径。

        Returns:
            numpy.ndarray: 医学图像的数组表示。

        """
        sitkImage = sitk.ReadImage(img_path)

        win_center = self.win_center
        win_width = self.win_width
        depth = self.depth
        min_num = int(win_center - win_width / 2.0)
        max_num = int(win_center + win_width / 2.0)

        intensityWindowing = sitk.IntensityWindowingImageFilter()
        intensityWindowing.SetOutputMaximum(max_num)
        intensityWindowing.SetOutputMinimum(min_num)

        # sitkImage = resampleSize(sitkImage, depth)
        sitkImage = intensityWindowing.Execute(sitkImage)
        data = sitk.GetArrayFromImage(sitkImage)
        return data

    def __len__(self):
        return len(self.file_list)


def resampleSpacing(sitkImage, newspace=(1, 1, 1)):
    """
    对输入的SimpleITK图像进行重采样，使其具有新的像素间距。

    Args:
        sitkImage (sitk.Image): 待重采样的SimpleITK图像。
        newspace (tuple, optional): 新的像素间距，默认为(1, 1, 1)。

    Returns:
        sitk.Image: 重采样后的SimpleITK图像。

    """
    euler3d = sitk.Euler3DTransform()
    xsize, ysize, zsize = sitkImage.GetSize()
    xspacing, yspacing, zspacing = sitkImage.GetSpacing()
    origin = sitkImage.GetOrigin()
    directions = sitkImage.GetDirection()
    new_size = (
        int(xsize * xspacing / newspace[0]), int(ysize * yspacing / newspace[1]), int(zsize * zspacing / newspace[2]))
    sitkImage = sitk.Resample(
        sitkImage, new_size, euler3d, sitk.sitkNearestNeighbor, origin, newspace, directions)
    return sitkImage


def resampleSize(sitkImage, depth):
    """
    对输入的SimpleITK图像进行重采样，改变其在Z轴上的切片数量。

    Args:
        sitkImage (sitk.Image): 待重采样的SimpleITK图像。
        depth (int): 重采样后Z轴上的切片数量。

    Returns:
        sitk.Image: 重采样后的SimpleITK图像。

    """
    euler3d = sitk.Euler3DTransform()
    xsize, ysize, zsize = sitkImage.GetSize()
    xspacing, yspacing, zspacing = sitkImage.GetSpacing()
    new_sapcing_z = zspacing / (depth / float(zsize))

    origin = sitkImage.GetOrigin()
    directions = sitkImage.GetDirection()

    new_size = (xsize, ysize, int(zsize * zspacing / new_sapcing_z))
    new_space = (xspacing, yspacing, new_sapcing_z)
    sitkImage = sitk.Resample(
        sitkImage, new_size, euler3d, sitk.sitkNearestNeighbor, origin, new_space, directions)
    return sitkImage


if __name__ == '__main__':
    dataset_root = 'data/BHSD'
    from Medical3DSeg.transforms import transform as T

    transforms = [
        T.RandomFlip3D(flip_axis=1),
        T.RandomRotation3D(degrees=10),
    ]
    dataset = MedicalDataset(
        dataset_root=dataset_root,
        transforms=transforms,
        num_classes=5,
    )
    for i in range(len(dataset)):
        img, label = dataset[i]
        print(img.shape, label.shape)
        print(type(img), type(label))
        break
    print(len(dataset))
