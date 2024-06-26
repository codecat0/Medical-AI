#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :transform.py
@Author :CodeCat
@Date   :2024/6/3 13:52
"""
from typing import List, Tuple, Union, Callable
import math
import random
import numpy as np
import numbers
import collections
import numpy as np
from Medical3DSeg.transforms import functional as F


class Compose:
    def __init__(self, transforms, isndhw=True, use_std=False):
        if not isinstance(transforms, list):
            raise TypeError("transforms must be a list of transforms")
        self.transforms = transforms
        self.isndhw = isndhw
        self.use_std = use_std

    def __call__(self, img, label=None, isndhw=True):
        """
        对输入的图像进行一系列的数据增强操作。

        Args:
            img (np.ndarray): 输入的图像，形状为[H, W, D]，其中H、W、D分别为图像的高、宽、通道数。
            label (np.ndarray, optional): 输入图像的标签，形状为[N]，其中N为标签数量。默认为None。
            isndhw (bool, optional): 是否将处理后的图像扩展一个维度，使其形状变为[1, D, H, W]。默认为True。

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: 返回一个元组，包含处理后的图像和标签。
            处理后的图像形状为[1, D, H, W]或[D, H, W]，标签形状为[N]或None。

        """
        mean = np.mean(img)
        std = np.std(img)

        for op in self.transforms:
            outputs = op(img, label)
            img = outputs[0]
            if len(outputs) == 2:
                label = outputs[1]

        if isndhw:
            img = np.expand_dims(img, axis=0)

        if (not self.use_std) and img.max() > 0:
            img = img / img.max()
        else:
            img = (img - mean) / (std + 1e-10)

        return img, label


class Resize3D:
    """Resize the input numpy ndarray to the given size. """

    def __init__(self, size, order=1):
        if isinstance(size, int):
            self.size = size
        elif isinstance(size, collections.abc.Iterable) and len(size) == 3:
            if type(size) == list:
                size = tuple(size)
            self.size = size
        else:
            raise ValueError('Unknown inputs for size: {}'.format(size))
        self.order = order
        super().__init__()

    def __call__(self, img, label=None):
        """
        对输入的3D图像和标签进行大小调整。

        Args:
            img (ndarray): 输入的3D图像，形状为(D, H, W)。
            label (Optional[ndarray], optional): 输入的3D标签图像，形状为(D, H, W)。默认为None。

        Returns:
            Tuple[ndarray, Optional[ndarray]]: 返回调整大小后的3D图像和标签图像。
                如果标签图像存在，则返回(调整后的3D图像, 调整后的3D标签图像)；
                否则返回(调整后的3D图像, None)。

        """
        img = F.resize_3d(img, self.size, self.order)
        if label is not None:
            label = F.resize_3d(label, self.size, 0)
        return img, label


class RandomRotation3D:
    """Rotate the image by angle."""

    def __init__(self, degrees, rotate_planes=[[0, 1], [0, 2], [1, 2]]):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError(
                    "If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError(
                    "If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.rotate_planes = rotate_planes

        super().__init__()

    def get_params(self, degrees):
        angle = random.uniform(degrees[0], degrees[1])
        r_plane = self.rotate_planes[random.randint(
            0, len(self.rotate_planes) - 1)]

        return angle, r_plane

    def __call__(self, img, label=None):
        """
        对输入的图像和标签进行三维旋转操作。

        Args:
            img (np.ndarray): 输入的三维图像数据，形状为 (D, H, W)。
            label (Optional[np.ndarray], optional): 输入的图像标签数据，形状与图像数据相同。默认为None。

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: 旋转后的三维图像数据和标签数据，形状与输入时相同。

        """

        angle, r_plane = self.get_params(self.degrees)
        img = F.rotate_3d(img, r_plane, angle)
        if label is not None:
            label = F.rotate_3d(label, r_plane, angle)
        return img, label


class RandomQuarterTurn3D:
    """ Rotate an 3D image 90 degrees with a certain probability."""

    def __init__(self, prob=0.5, rotate_planes=[[0, 1], [0, 2], [1, 2]]):
        self.rotate_planes = rotate_planes
        self.prob = prob
        super().__init__()

    def get_params(self):
        r_plane = self.rotate_planes[random.randint(
            0, len(self.rotate_planes) - 1)]
        return r_plane

    def __call__(self, img, label=None):
        """
        对输入的图像和标签进行随机三维旋转操作。

        Args:
            img (np.ndarray): 输入的图像数据，形状为 (D, H, W) 的三维数组。
            label (Optional[np.ndarray], optional): 输入的图像标签数据，形状与 img 相同。默认为 None。

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: 经过三维旋转后的图像数据和标签数据，形状与输入时相同。

        """

        if random.random() < self.prob:
            r_plane = self.get_params()
            img = F.rotate_3d(img, r_plane, 90)
            if label is not None:
                label = F.rotate_3d(label, r_plane, 90)
        return img, label


class RandomFlip3D:
    """Flip an 3D image with a certain probability.
    Args:
        prob (float, optional): A probability of vertical flipping. Default: 0.1.
    """

    def __init__(self, prob=0.5, flip_axis=[0, 1, 2]):
        """
        init
        """
        self.prob = prob
        self.flip_axis = flip_axis

        super().__init__()

    def __call__(self, img, label=None):
        """
        对输入的图像进行翻转操作。

        Args:
            img (np.ndarray): 输入的图像数据，形状为 (D, H, W)。
            label (np.ndarray, optional): 输入的图像对应的标签数据，形状为 (D, H, W)。默认为 None。

        Returns:
            Tuple[np.ndarray, Union[None, np.ndarray]]: 返回翻转后的图像数据和对应的标签数据，标签数据只有在输入时不为 None 时才返回。

        """
        if isinstance(self.flip_axis, (tuple, list)):
            flip_axis = self.flip_axis[random.randint(0,
                                                      len(self.flip_axis) - 1)]
        else:
            flip_axis = self.flip_axis

        if random.random() < self.prob:
            img = F.flip_3d(img, axis=flip_axis)
            if label is not None:
                label = F.flip_3d(label, axis=flip_axis)
        return img, label


class RandomResizedCrop3D:
    """
    先Crop再Resize至预设尺寸
    scale: 切出cube的体积与原图体积的比值范围
    ratio: 切出cube的每一边长的抖动范围
    size:  resize的目标尺寸
    interpolation: [1-5]， skimage.zoom的order数。注意分割模式下label的order统一为0
    pre_crop: bool，如果为True，则先切一个目标尺寸左右的cube，再resize，通常用于滑窗模式；
                    如果为False，则从原图上扣一个与原图接近的cube，再resize至目标尺寸
    nonzero_mask，如果为True，则只在label mask有效(非0)区域内进行滑窗
                  如果为False，则在image整个区域内进行滑窗
    """

    def __init__(self, size, scale=(0.8, 1.2), ratio=(3. / 4., 4. / 3.), \
                 interpolation=1, pre_crop=False, nonzero_mask=False):
        """
        init
        """
        if isinstance(size, (tuple, list)):
            assert len(size) == 3, \
                "Size must contain THREE number when it is a tuple or list, got {}.".format(len(size))
            self.size = size
        elif isinstance(size, int):
            self.size = (size, size, size)
        else:
            print("Size must be a list or tuple, got {}.".format(type(size)))

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.pre_crop = pre_crop
        self.nonzero_mask = nonzero_mask

        super().__init__()

    def get_params(self, img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (numpy ndarray): Image to be cropped. d, h, w
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        params_ret = collections.namedtuple('params_ret',
                                            ['i', 'j', 'k', 'd', 'h', 'w'])
        for attempt in range(10):
            volume = img.shape[0] * img.shape[1] * img.shape[2]
            target_volume = random.uniform(*scale) * volume
            aspect_ratio = random.uniform(*ratio)

            d = int(round((target_volume * aspect_ratio) ** (1 / 3)))
            h = int(round((target_volume / aspect_ratio) ** (1 / 3)))
            w = img.shape[2]

            if random.random() < 0.5:
                d, h, w = random.sample([d, h, w], k=3)

            if w <= img.shape[2] and h <= img.shape[1] and d <= img.shape[0]:
                i = random.randint(0, img.shape[0] - d)
                j = random.randint(0, img.shape[1] - h)
                k = random.randint(0, img.shape[2] - w)
                return params_ret(i, j, k, d, h, w)

        # Fallback
        w = min(img.shape[0], img.shape[1], img.shape[2])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        k = (img.shape[2] - w) // 2
        return params_ret(i, j, k, w, w, w)

    def pre_crop_util(self, img, label=None):
        """
        pre crop逻辑
        """
        if self.pre_crop:
            crop_size = (np.random.uniform(low=self.scale[0], high=self.scale[1], size=3) * \
                         self.size).round().astype("int")

            if self.nonzero_mask:
                mask_voxel_coords = np.where(label != 0)
                minzidx = int(np.min(mask_voxel_coords[0]))
                maxzidx = int(np.max(mask_voxel_coords[0])) + 1
                minyidx = int(np.min(mask_voxel_coords[1]))
                maxyidx = int(np.max(mask_voxel_coords[1])) + 1
                minxidx = int(np.min(mask_voxel_coords[2]))
                maxxidx = int(np.max(mask_voxel_coords[2])) + 1

                masked_shape = np.array(
                    [maxzidx - minzidx, maxyidx - minyidx, maxxidx - minxidx])
                crop_z, crop_y, crop_x = np.minimum(masked_shape, crop_size)
                z_start = np.random.randint(masked_shape[0] - crop_z +
                                            1) + minzidx
                y_start = np.random.randint(masked_shape[1] - crop_y +
                                            1) + minyidx
                x_start = np.random.randint(masked_shape[2] - crop_x +
                                            1) + minxidx

                z_end = z_start + crop_z
                y_end = y_start + crop_y
                x_end = x_start + crop_x
            else:
                crop_z, crop_y, crop_x = np.minimum(img.shape[:3], crop_size)
                z_start = np.random.randint(img.shape[0] - crop_z + 1)
                y_start = np.random.randint(img.shape[1] - crop_y + 1)
                x_start = np.random.randint(img.shape[2] - crop_x + 1)

                z_end = z_start + crop_z
                y_end = y_start + crop_y
                x_end = x_start + crop_x

            img = img[z_start:z_end, y_start:y_end, x_start:x_end]
            if label is not None:
                label = label[z_start:z_end, y_start:y_end, x_start:x_end]

        return img, label

    def __call__(self, img, label=None):
        """
        对输入的3D图像和标签进行随机裁剪和缩放。

        Args:
            img (torch.Tensor): 输入的3D图像，形状为(D, H, W)，其中D为深度，H为高度，W为宽度。
            label (Optional[torch.Tensor], optional): 输入的3D图像标签，形状为(D, H, W)，默认为None。

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: 返回一个元组，包含裁剪和缩放后的3D图像和标签。
                - 裁剪和缩放后的3D图像，形状为(new_D, new_H, new_W)，其中new_D, new_H, new_W分别为裁剪和缩放后的深度、高度和宽度。
                - 如果传入了标签，则同时返回裁剪和缩放后的标签，形状为(new_D, new_H, new_W)；否则返回None。

        """

        img, label = self.pre_crop_util(img, label)
        i, j, k, d, h, w = self.get_params(img, self.scale, self.ratio)
        img = F.resized_crop_3d(img, i, j, k, d, h, w, self.size,
                                self.interpolation)
        if label is not None:
            label = F.resized_crop_3d(label, i, j, k, d, h, w, self.size, 0)

        return img, label


class BinaryMaskToConnectComponent:
    """Got the connect compoent from binary mask
    Args:
        minimum_volume (int, default=0): The minimum volume of the connected component to be retained
    """

    def __init__(self, minimum_volume=0):
        """
        resize
        """
        self.minimum_volume = minimum_volume
        super().__init__()

    def __call__(self, pred, label=None):
        """
        对输入的预测结果和标签进行连通域提取操作。

        Args:
            pred (numpy.ndarray): 输入的预测结果，形状为 (H, W)，数据类型为 float32 或 uint8。
            label (Optional[numpy.ndarray], optional): 输入的标签，形状为 (H, W)，数据类型为 int32 或 int64。默认为 None。

        Returns:
            Tuple[numpy.ndarray, Optional[numpy.ndarray]]: 返回经过连通域提取操作后的预测结果和标签，形状与输入相同。如果标签为 None，则返回 None。

        """

        pred = F.extract_connect_compoent(pred, self.minimum_volume)
        if label is not None:
            label = F.extract_connect_compoent(label, self.minimum_volume)
        return pred, label


class TopkLargestConnectComponent:
    """Keep topk largest connect component sorted by volume nums, remove others.
    Args:
        k (int, default=1): k
    """

    def __init__(self, k=1):
        """
        resize
        """
        self.k = k
        super().__init__()

    def __call__(self, pred, label=None):
        """
        Args:
            pred (numpy ndarray): Image to be scaled.
            label (numpy ndarray) : Label to be scaled
        Returns:
            numpy ndarray: Rescaled image.
            numpy ndarray: Rescaled label.
        """
        pred = F.extract_connect_compoent(pred)
        pred[pred > self.k] = 0
        return pred, label


class GaussianNoiseTransform:
    """
    对原image增加高斯噪声，返回增加噪声后的image
    noise_variance: 高斯噪声的variance参数设置
    p_per_sample: 对原image增加噪声的概率
    p_per_channel:  对image的每个channel产生噪声的概率
    per_channel: 是否对不同channel进行噪声增强
    """

    def __init__(self,
                 noise_variance=(0, 0.1),
                 p_per_sample=1,
                 p_per_channel: float = 1,
                 per_channel: bool = False):
        self.p_per_sample = p_per_sample
        self.noise_variance = noise_variance
        self.p_per_channel = p_per_channel
        self.per_channel = per_channel

    def __call__(self, img, label=None):
        """
        对输入的图像和标签进行高斯噪声增强，并返回增强后的图像和标签。

        Args:
            img (np.ndarray): 待增强的图像，数据类型为numpy数组。
            label (np.ndarray, optional): 待增强的标签，数据类型为numpy数组。默认为None。

        Returns:
            Tuple[np.ndarray, np.ndarray]: 返回一个元组，包含增强后的图像和标签，数据类型均为numpy数组。

        """
        if np.random.uniform() < self.p_per_sample:
            img = F.augment_gaussian_noise(img, self.noise_variance,
                                           self.p_per_channel, self.per_channel)
        return img, label


class GaussianBlurTransform:
    """
    对原image增加高斯模糊增强
    blur_sigma: 高斯模糊的sigma参数设置
    different_sigma_per_channel: 不同channel进行不同sigma的设置
    different_sigma_per_axis:  不同axis进行不同sigma的设置
    p_isotropic: isotropic的概率
    p_per_channel：每次对channel上进行高斯模糊增强概率
    p_per_sample: 对原image进行该项数据增强的概率
    """

    def __init__(self,
                 blur_sigma: Tuple[float, float] = (1, 5),
                 different_sigma_per_channel: bool = True,
                 different_sigma_per_axis: bool = False,
                 p_isotropic: float = 0,
                 p_per_channel: float = 1,
                 p_per_sample: float = 1):
        self.p_per_sample = p_per_sample
        self.different_sigma_per_channel = different_sigma_per_channel
        self.p_per_channel = p_per_channel
        self.blur_sigma = blur_sigma
        self.different_sigma_per_axis = different_sigma_per_axis
        self.p_isotropic = p_isotropic

    def __call__(self, img, label=None):
        """
        对输入图像进行高斯模糊增强。

        Args:
            img (np.ndarray): 待增强的图像，数据类型为numpy数组。
            label (np.ndarray, optional): 图像的标签，数据类型为numpy数组。默认为None。

        Returns:
            Tuple[np.ndarray, np.ndarray]: 返回一个元组，包含增强后的图像和标签，数据类型均为numpy数组。

        """

        if np.random.uniform() < self.p_per_sample:
            img = F.augment_gaussian_blur(
                img,
                self.blur_sigma,
                self.different_sigma_per_channel,
                self.p_per_channel,
                different_sigma_per_axis=self.different_sigma_per_axis,
                p_isotropic=self.p_isotropic)
        return img, label


class BrightnessMultiplicativeTransform:
    """
    对原image明亮度进行变化
    multiplier_range: 变化范围参数
    per_channel: 是否对每个channel进行变化
    p_per_sample: 对原image进行该项数据增强的概率
    """

    def __init__(self,
                 multiplier_range=(0.5, 2),
                 per_channel=True,
                 p_per_sample=1):
        self.p_per_sample = p_per_sample
        self.multiplier_range = multiplier_range
        self.per_channel = per_channel

    def __call__(self, img, label=None):
        """
        对输入图像进行亮度增强。

        Args:
            img (np.ndarray): 待增强的图像，数据类型为numpy数组。
            label (np.ndarray, optional): 图像的标签，数据类型为numpy数组。默认为None。

        Returns:
            Tuple[np.ndarray, np.ndarray]: 返回一个元组，包含增强后的图像和标签，数据类型均为numpy数组。

        """

        if np.random.uniform() < self.p_per_sample:
            img = F.augment_brightness_multiplicative(
                img, self.multiplier_range, self.per_channel)
        return img, label


class ContrastAugmentationTransform:
    """
    对原image对比度进行变化实现对比度上的图像增强
    contrast_range: 对比度变化范围参数
    preserve_range: 保护参数
    per_channel: 对原image每个channel进行该项变化
    p_per_sample: 对原image进行该项数据增强的概率
    p_per_channel: 对原image每个channel上进行该项数据增强的概率
    """

    def __init__(
            self,
            contrast_range: Union[Tuple[float, float], Callable[[], float]] = (
                    0.75, 1.25),
            preserve_range: bool = True,
            per_channel: bool = True,
            p_per_sample: float = 1,
            p_per_channel: float = 1):
        self.p_per_sample = p_per_sample
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel

    def __call__(self, img, label=None):
        """
        对输入图像进行对比度增强。

        Args:
            img (np.ndarray): 待增强的图像，数据类型为numpy数组。
            label (np.ndarray, optional): 图像的标签，数据类型为numpy数组。默认为None。

        Returns:
            Tuple[np.ndarray, np.ndarray]: 返回一个元组，包含增强后的图像和标签，数据类型均为numpy数组。

        """

        if np.random.uniform() < self.p_per_sample:
            img = F.augment_contrast(
                img,
                contrast_range=self.contrast_range,
                preserve_range=self.preserve_range,
                per_channel=self.per_channel,
                p_per_channel=self.p_per_channel)
        return img, label


class SimulateLowResolutionTransform:
    """
    对原image进行随机的分辨率压缩的数据增强方法
    zoom_range: 分辨率缩放范围
    per_channel: 对原image每个channel进行该项变化
    p_per_channel: 对原image每个channel上进行该项数据增强的概率
    channels: 指定进行该项变化的channels
    order_downsample：底层进行zoom时候的order参数
    order_upsample：底层进行zoom时候的order参数
    p_per_sample: 对原image进行该项数据增强的概率
    ignore_axes： 是否忽略axes
    """

    def __init__(self,
                 zoom_range=(0.5, 1),
                 per_channel=False,
                 p_per_channel=1,
                 channels=None,
                 order_downsample=1,
                 order_upsample=0,
                 p_per_sample=1,
                 ignore_axes=None):
        self.order_upsample = order_upsample
        self.order_downsample = order_downsample
        self.channels = channels
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
        self.p_per_sample = p_per_sample
        self.zoom_range = zoom_range
        self.ignore_axes = ignore_axes

    def __call__(self, img, label=None):
        """
        对输入的图像行随机的分辨率压缩的数据增强，并返回增强后的图像和标签。

        Args:
            img (np.ndarray): 待增强的图像，数据类型为numpy数组。
            label (np.ndarray, optional): 图像的标签，数据类型为numpy数组。默认为None。

        Returns:
            Tuple[np.ndarray, np.ndarray]: 返回一个元组，包含增强后的图像和标签，数据类型均为numpy数组。

        """
        if np.random.uniform() < self.p_per_sample:
            img = F.augment_linear_downsampling_scipy(
                img,
                zoom_range=self.zoom_range,
                per_channel=self.per_channel,
                p_per_channel=self.p_per_channel,
                channels=self.channels,
                order_downsample=self.order_downsample,
                order_upsample=self.order_upsample,
                ignore_axes=self.ignore_axes)
        return img, label


class GammaTransform:
    """
    对原image进行gamma数据增强方法
    gamma_range: gamma范围
    invert_image: 是否进行灰度值反转
    per_channel: 是否对每个channel进行该项增强
    retain_stats: 是否保留状态
    p_per_sample: 对原image进行该项数据增强的概率
    """

    def __init__(self,
                 gamma_range=(0.5, 2),
                 invert_image=False,
                 per_channel=False,
                 retain_stats: Union[bool, Callable[[], bool]]=False,
                 p_per_sample=1):
        self.p_per_sample = p_per_sample
        self.retain_stats = retain_stats
        self.per_channel = per_channel
        self.gamma_range = gamma_range
        self.invert_image = invert_image

    def __call__(self, img, label=None):
        """
        对输入图像进行gamma增强。

        Args:
            img (np.ndarray): 待增强的图像，数据类型为numpy数组。
            label (np.ndarray, optional): 图像的标签，数据类型为numpy数组。默认为None。

        Returns:
            Tuple[np.ndarray, np.ndarray]: 返回一个元组，包含增强后的图像和标签，数据类型均为numpy数组。

        """

        if np.random.uniform() < self.p_per_sample:
            img = F.augment_gamma(
                img,
                self.gamma_range,
                self.invert_image,
                per_channel=self.per_channel,
                retain_stats=self.retain_stats)
        return img, label


class MirrorTransform:
    """
    对原image进行空间上进行镜像翻转操作
    axes: 进行翻转的axes
    p_per_sample: 对原image进行该项数据增强的概率
    """

    def __init__(self, axes=(0, 1, 2), p_per_sample=1):
        self.p_per_sample = p_per_sample
        self.axes = axes
        if max(axes) > 2:
            raise ValueError(
                "MirrorTransform now takes the axes as the spatial dimensions. What previously was "
                "axes=(2, 3, 4) to mirror along all spatial dimensions of a 5d tensor (b, c, x, y, z) "
                "is now axes=(0, 1, 2). Please adapt your scripts accordingly.")

    def __call__(self, img, label=None):
        """
        对输入的图像和标签进行镜像翻转增强。

        Args:
            img (np.ndarray): 待增强的图像，数据类型为numpy数组。
            label (np.ndarray, optional): 图像的标签，数据类型为numpy数组。默认为None。

        Returns:
            Tuple[np.ndarray, np.ndarray]: 返回一个元组，包含增强后的图像和标签，数据类型均为numpy数组。

        """
        if np.random.uniform() < self.p_per_sample:
            sample_seg = None
            if label is not None:
                sample_seg = label
            ret_val = F.augment_mirroring(img, label, axes=self.axes)
            img = ret_val[0]
            if label is not None:
                label = ret_val[1]

        return img, label


class ResizeRangeScaling:
    """
    Resize the long side of an image into a range, and then scale the other side proportionally.

    Args:
        min_value (int, optional): The minimum value of long side after resize. Default: 400.
        max_value (int, optional): The maximum value of long side after resize. Default: 600.
    """

    def __init__(self,
                 min_scale_factor=0.85,
                 max_scale_factor=1.25,
                 interpolation=3,
                 p_per_sample=1):
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.p_per_sample = p_per_sample
        self.interpolation = interpolation

    def __call__(self, img, label=None):
        """
        对输入的3D图像和标签进行随机缩放增强。

        Args:
            img (np.ndarray): 待增强的3D图像，数据类型为numpy数组，形状为(depth, width, height)。
            label (np.ndarray, optional): 图像的标签，数据类型为numpy数组，形状为(depth, width, height)。默认为None。

        Returns:
            Tuple[np.ndarray, np.ndarray]: 返回一个元组，包含增强后的3D图像和标签，数据类型均为numpy数组，形状均为(new_depth, new_width, new_height)。

        """
        scale_factor = 1
        if np.random.uniform() < self.p_per_sample:
            scale_factor = np.random.uniform(self.min_scale_factor,
                                             self.max_scale_factor)
        d, w, h = img.shape
        new_d = d * scale_factor
        new_w = w * scale_factor
        new_h = h * scale_factor
        img = F.resize_3d(img, [new_d, new_w, new_h], self.interpolation)
        if label is not None:
            label = F.resize_3d(label, [new_d, new_w, new_h], 0)

        return img, label


class RandomPaddingCrop:
    """
    Crop a sub-image from a raw image and annotation image randomly. If the target cropping size
    is larger than original image, then the bottom-right padding will be added.

    Args:
        crop_size (tuple, optional): The target cropping size. Default: (512, 512).
        im_padding_value (list, optional): The padding value of raw image.
            Default: [127.5, 127.5, 127.5].
        label_padding_value (int, optional): The padding value of annotation image. Default: 255.

    Raises:
        TypeError: When crop_size is neither list nor tuple.
        ValueError: When the length of crop_size is not 2.
    """

    def __init__(self,
                 crop_size=(512, 512, 512),
                 im_padding_value=0,
                 label_padding_value=0):
        if isinstance(crop_size, list) or isinstance(crop_size, tuple):
            if len(crop_size) != 3:
                raise ValueError(
                    'Type of `crop_size` is list or tuple. It should include 3 elements, but it is {}'
                    .format(crop_size))
        else:
            raise TypeError(
                "The type of `crop_size` is invalid. It should be list or tuple, but it is {}"
                .format(type(crop_size)))
        self.crop_size = crop_size
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, img, label=None):
        """
        对输入的3D图像和标签进行随机裁剪。

        Args:
            img (np.ndarray): 待裁剪的3D图像，数据类型为numpy数组，形状为(depth, height, width)。
            label (np.ndarray, optional): 图像的标签，数据类型为numpy数组，形状为(depth, height, width)。默认为None。

        Returns:
            Tuple[np.ndarray, np.ndarray]: 返回一个元组，包含裁剪后的3D图像和标签，数据类型均为numpy数组，形状均为(crop_depth, crop_height, crop_width)。

        """

        if isinstance(self.crop_size, int):
            crop_depth = self.crop_size
            crop_width = self.crop_size
            crop_height = self.crop_size
        else:
            crop_depth = self.crop_size[0]
            crop_width = self.crop_size[1]
            crop_height = self.crop_size[2]

        img_depth = img.shape[0]
        img_height = img.shape[1]
        img_width = img.shape[2]

        if img_height == crop_height and img_width == crop_width and img_depth == crop_depth:
            return img, label
        else:
            pad_depth = max(crop_depth - img_depth, 0)
            pad_height = max(crop_height - img_height, 0)
            pad_width = max(crop_width - img_width, 0)
            if pad_height > 0 or pad_width > 0 or pad_depth > 0:
                img = np.pad(img, ((0, pad_depth), (0, pad_height),
                                   (0, pad_width)))
                if label is not None:
                    label = np.pad(label, ((0, pad_depth), (0, pad_height),
                                           (0, pad_width)))

                img_depth = img.shape[0]
                img_height = img.shape[1]
                img_width = img.shape[2]

            if crop_depth > 0 and crop_height > 0 and crop_width > 0:
                d_off = np.random.randint(img_depth - crop_depth + 1)
                h_off = np.random.randint(img_height - crop_height + 1)
                w_off = np.random.randint(img_width - crop_width + 1)
                img = img[d_off:(d_off + crop_depth), h_off:(
                    crop_height + h_off), w_off:(w_off + crop_width)]
                if label is not None:
                    label = label[d_off:(d_off + crop_depth), h_off:(
                        crop_height + h_off), w_off:(w_off + crop_width)]
        return img, label