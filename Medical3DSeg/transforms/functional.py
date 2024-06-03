#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :functional.py
@Author :CodeCat
@Date   :2024/6/3 10:43
"""
from typing import List, Tuple, Union, Callable
import collections
import numbers
import random

import numpy as np
import scipy
import scipy.ndimage
import SimpleITK as sitk
from scipy.ndimage.filters import gaussian_filter, gaussian_gradient_filter
from skimage.transform import resize


def _is_numpy_image(img):
    """
    判断输入的对象是否为 NumPy 类型的图像数组。

    Args:
        img (np.ndarray): 待判断的对象。

    Returns:
        bool: 若输入的对象为 NumPy 类型的二维或三维数组，则返回 True；否则返回 False。

    """
    return isinstance(img, np.ndarray) and len(img.shape) in (2, 3)


def resize_3d(image, size, order=1):
    """
    将3D图像调整至指定大小。

    Args:
        image (ndarray): 输入的3D图像。
        size (int or tuple of length 3): 输出图像的尺寸。如果传入单个整数，则将其用作所有三个维度的长度。
        order (int, optional): scipy.zoom的期望阶数。默认为1。

    Returns:
        resized_image (ndarray): 调整大小后的3D图像。

    Raises:
        TypeError: 如果输入不是numpy数组，则引发TypeError。
        ValueError: 如果输入大小不是整数或长度为3的元组，则引发ValueError。
    """
    if not _is_numpy_image(image):
        raise TypeError('Input must be a numpy array. Got {}'.format(type(image)))

    if not (isinstance(size, int) or len(size) == 3):
        raise ValueError('Input size must be an integer or a tuple of length 3. Got {}'.format(type(size)))

    d, h, w = image.shape[0], image.shape[1], image.shape[2]

    if isinstance(size, int):
        if min(d, h, w) == size:
            return image

        nw = int(size * w / min(d, h, w))
        nh = int(size * h / min(d, h, w))
        nd = int(size * d / min(d, h, w))

    else:
        nw, nh, nd = size[2], size[1], size[0]

    resize_factor = np.array([nd / d, nh / h, nw / w])
    output = scipy.ndimage.zoom(
        image, resize_factor, mode='nearest', order=order
    )
    return output


def crop_3d(image, i, j, k, d, h, w):
    """
    裁剪3D图像。

    Args:
        image (np.ndarray): 待裁剪的3D图像。
        i (int): 裁剪区域的起始深度坐标。
        j (int): 裁剪区域的起始高度坐标。
        k (int): 裁剪区域的起始宽度坐标。
        d (int): 裁剪区域的深度大小。
        h (int): 裁剪区域的高度大小。
        w (int): 裁剪区域的宽度大小。

    Returns:
        np.ndarray: 裁剪后的3D图像。

    Raises:
        TypeError: 如果输入不是numpy数组，则引发TypeError。
    """
    if not _is_numpy_image(image):
        raise TypeError('Input must be a numpy array. Got {}'.format(type(image)))

    return image[i:i + d, j: j + h, k:j + w]


def flip_3d(image, axis):
    """
    将3D图像沿指定轴翻转。

    Args:
        image (np.ndarray): 待翻转的3D图像。
        axis (int): 翻转的轴，取值为0, 1, 2，分别对应深度、高度、宽度。

    Returns:
        np.ndarray: 翻转后的3D图像。

    """
    image = np.flip(image, axis=axis)
    return image


def rotate_3d(image, r_plane, angle, order=1, cval=0):
    """
    对3D图像进行旋转。

    Args:
        image (np.ndarray): 待旋转的3D图像。
        r_plane (tuple): 旋转平面，元组中包含两个整数，表示要旋转的轴，如(0, 1)表示在深度和高度的平面上旋转。
        angle (float): 旋转角度，单位为度。
        order (int, optional): 插值方式。0表示最近邻插值，1表示双线性插值，2表示双三次插值等。默认为1。
        cval (float, optional): 当需要插值时使用的值。默认为0。

    Returns:
        np.ndarray: 旋转后的3D图像。

    """
    image = scipy.ndimage.rotate(
        image, angle=angle, axes=r_plane, order=order, cval=cval, reshape=False
    )
    return image


def resized_crop_3d(image, i, j, k, d, h, w, size, interpolation):
    """
    对3D图像进行裁剪和缩放。

    Args:
        image (np.ndarray): 待裁剪和缩放的3D图像。
        i (int): 裁剪区域的起始深度坐标。
        j (int): 裁剪区域的起始高度坐标。
        k (int): 裁剪区域的起始宽度坐标。
        d (int): 裁剪区域的深度大小。
        h (int): 裁剪区域的高度大小。
        w (int): 裁剪区域的宽度大小。
        size (int or tuple of length 3): 输出图像的尺寸。如果传入单个整数，则将其用作所有三个维度的长度。
        interpolation (int): 缩放时使用的插值方式。0表示最近邻插值，1表示双线性插值，2表示双三次插值等。

    Returns:
        np.ndarray: 裁剪并缩放后的3D图像。

    Raises:
        AssertionError: 如果输入不是numpy数组，则引发AssertionError。
    """
    assert _is_numpy_image(image), 'img should be numpy image'
    image = crop_3d(image, i, j, k, d, h, w)
    image = resize_3d(image, size, order=interpolation)
    return image


def extract_connect_components(binary_mask, minimum_volume=0):
    """
    从二值化掩膜中提取连通组件。

    Args:
        binary_mask (np.ndarray): 二值化掩膜，只包含0和1两个值。
        minimum_volume (int, optional): 连通组件的最小体积。默认值为0。

    Returns:
        np.ndarray: 提取出的连通组件掩膜，每个连通组件被标记为不同的整数。

    Raises:
        AssertionError: 如果输入的掩膜不是二值化的，将引发断言错误。

    """
    assert len(np.unique(binary_mask)) < 3, \
        'Only binary masks are supported. Got mask with {}'.format(np.unique(binary_mask).tolist())

    instance_mask = sitk.GetArrayFromImage(
        sitk.RelabelComponent(
            sitk.ConnectedComponent(sitk.GetImageFromArray(binary_mask)),
            minimumObjectSize=minimum_volume
        )
    )
    return instance_mask


def augment_gaussian_noise(data_sample, noise_variance=(0, 0.1), p_per_channel=1, per_channel=False):
    """
    向数据样本添加高斯噪声。

    Args:
        data_sample (np.ndarray): 输入的数据样本，形状为 (channels, height, width) 或 (height, width, channels)。
        noise_variance (tuple[float, float], optional): 噪声方差的范围，即 (min_variance, max_variance)。默认值为 (0, 0.1)。
        p_per_channel (float, optional): 每个通道添加噪声的概率。默认值为 1，表示每个通道都会添加噪声。
        per_channel (bool, optional): 是否为每个通道添加不同方差的噪声。如果为 True，则每个通道将独立地选择噪声方差。默认值为 False。

    Returns:
        np.ndarray: 添加了噪声的数据样本，形状与输入数据样本相同。

    """
    if not per_channel:
        variance = noise_variance[0] if noise_variance[0] == noise_variance[1] else np.random.uniform(
            noise_variance[0], noise_variance[1]
        )
    else:
        variance = None

    for c in range(data_sample.shape[0]):
        if np.random.uniform() < p_per_channel:
            variance_here = variance if variance is not None else \
                noise_variance[0] if noise_variance[0] == noise_variance[1] else np.random.uniform(
                    noise_variance[0], noise_variance[1]
                )
            data_sample[c] = data_sample[c] + np.random.normal(0, variance_here, size=data_sample[c].shape)

    return data_sample


def augment_gaussian_blur(data_sample: np.ndarray,
                          sigma_range: Tuple[float, float],
                          per_channel: bool = True,
                          p_per_channel: float = 1,
                          different_sigma_per_axis: bool = False,
                          p_isotropic: float = 0) -> np.ndarray:
    """
       对输入的数据样本进行高斯模糊增强。

       Args:
           data_sample (np.ndarray): 输入的数据样本，形状为 (channels, height, width) 或 (height, width, channels)。
           sigma_range (Tuple[float, float]): 模糊程度sigma的取值范围，为包含两个元素的元组，表示sigma的最小值和最大值。
           per_channel (bool, optional): 是否对每个通道独立进行模糊增强。默认为True。
           p_per_channel (float, optional): 每个通道进行模糊增强的概率。默认为1，表示每个通道都会进行模糊增强。
           different_sigma_per_axis (bool, optional): 是否对每个维度（轴）独立进行模糊增强，并使用不同的sigma值。默认为False。
           p_isotropic (float, optional): 模糊增强为各向同性的概率。默认为0。

       Returns:
           np.ndarray: 进行了高斯模糊增强的数据样本，形状与输入数据样本相同。

       Raises:
           无

       """

    def get_range_val(value, rnd_type="uniform"):
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 2:
                if value[0] == value[1]:
                    n_val = value[0]
                else:
                    orig_type = type(value[0])
                    if rnd_type == "uniform":
                        n_val = random.uniform(value[0], value[1])
                    else:
                        n_val = random.normalvariate(value[0], value[1])
                    n_val = orig_type(n_val)
            elif len(value) == 1:
                n_val = value[0]
            else:
                raise RuntimeError(
                    "`value` must be a list/tuple with one or two elements.")
            return n_val
        else:
            return value

    if not per_channel:
        sigma = get_range_val(sigma_range) if ((not different_sigma_per_axis) or
                                               ((np.random.uniform() < p_isotropic) and
                                                different_sigma_per_axis)) \
            else [get_range_val(sigma_range) for _ in data_sample.shape[1:]]
    else:
        sigma = None
    for c in range(data_sample.shape[0]):
        if np.random.uniform() <= p_per_channel:
            if per_channel:
                sigma = get_range_val(sigma_range) if ((not different_sigma_per_axis) or
                                                       ((np.random.uniform() < p_isotropic) and
                                                        different_sigma_per_axis)) \
                    else [get_range_val(sigma_range) for _ in data_sample.shape[1:]]
            data_sample[c] = gaussian_filter(data_sample[c], sigma, order=0)
    return data_sample


def augment_brightness_multiplicative(data_sample,
                                      multiplier_range=(0.5, 2),
                                      per_channel=True):
    """
    对输入的数据样本进行亮度乘法增强。

    Args:
        data_sample (np.ndarray): 输入的数据样本，形状为 (channels, height, width) 或 (height, width, channels)。
        multiplier_range (Tuple[float, float], optional): 乘法因子的取值范围，为包含两个元素的元组，表示乘法因子的最小值和最大值。默认为 (0.5, 2)。
        per_channel (bool, optional): 是否对每个通道独立进行亮度乘法增强。默认为 True。

    Returns:
        np.ndarray: 进行了亮度乘法增强的数据样本，形状与输入数据样本相同。

    """
    multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1])
    if not per_channel:
        data_sample *= multiplier
    else:
        for c in range(data_sample.shape[0]):
            multiplier = np.random.uniform(multiplier_range[0],
                                           multiplier_range[1])
            data_sample[c] *= multiplier
    return data_sample


def augment_contrast(
        data_sample: np.ndarray,
        contrast_range: Union[Tuple[float, float], Callable[[], float]] = (0.75,
                                                                           1.25),
        preserve_range: bool = True,
        per_channel: bool = True,
        p_per_channel: float = 1) -> np.ndarray:
    """
    对输入的数据样本进行对比度增强。

    Args:
        data_sample (np.ndarray): 输入的数据样本，形状为 (channels, height, width) 或 (height, width, channels)。
        contrast_range (Union[Tuple[float, float], Callable[[], float]], optional): 对比度增强的取值范围，可以为元组或可调用对象。
            元组时，表示对比度增强的最小值和最大值；可调用对象时，每次调用返回一个对比度增强值。默认为 (0.75, 1.25)。
        preserve_range (bool, optional): 是否保持原始数据范围不变。默认为 True。
        per_channel (bool, optional): 是否对每个通道独立进行对比度增强。默认为 True。
        p_per_channel (float, optional): 每个通道进行对比度增强的概率。默认为 1。

    Returns:
        np.ndarray: 进行了对比度增强的数据样本，形状与输入数据样本相同。

    """
    if not per_channel:
        if callable(contrast_range):
            factor = contrast_range()
        else:
            if np.random.random() < 0.5 and contrast_range[0] < 1:
                factor = np.random.uniform(contrast_range[0], 1)
            else:
                factor = np.random.uniform(
                    max(contrast_range[0], 1), contrast_range[1])

        for c in range(data_sample.shape[0]):
            if np.random.uniform() < p_per_channel:
                mn = data_sample[c].mean()
                if preserve_range:
                    minm = data_sample[c].min()
                    maxm = data_sample[c].max()

                data_sample[c] = (data_sample[c] - mn) * factor + mn

                if preserve_range:
                    data_sample[c][data_sample[c] < minm] = minm
                    data_sample[c][data_sample[c] > maxm] = maxm
    else:
        for c in range(data_sample.shape[0]):
            if np.random.uniform() < p_per_channel:
                if callable(contrast_range):
                    factor = contrast_range()
                else:
                    if np.random.random() < 0.5 and contrast_range[0] < 1:
                        factor = np.random.uniform(contrast_range[0], 1)
                    else:
                        factor = np.random.uniform(
                            max(contrast_range[0], 1), contrast_range[1])

                mn = data_sample[c].mean()
                if preserve_range:
                    minm = data_sample[c].min()
                    maxm = data_sample[c].max()

                data_sample[c] = (data_sample[c] - mn) * factor + mn

                if preserve_range:
                    data_sample[c][data_sample[c] < minm] = minm
                    data_sample[c][data_sample[c] > maxm] = maxm
    return data_sample


def augment_linear_downsampling_scipy(data_sample,
                                      zoom_range=(0.5, 1),
                                      per_channel=True,
                                      p_per_channel=1,
                                      channels=None,
                                      order_downsample=1,
                                      order_upsample=0,
                                      ignore_axes=None):
    """
    对输入的数据样本进行线性下采样增强。

    Args:
        data_sample (np.ndarray): 输入的数据样本，形状为 (channels, height, width) 或 (height, width, channels)。
        zoom_range (Union[float, Tuple[float, float]], optional): 下采样率的取值范围，可以为单个浮点数或一个包含两个浮点数的元组。当为单个浮点数时，所有维度的下采样率相同；当为元组时，元组的每个元素对应一个维度的下采样率范围。默认为 (0.5, 1)。
        per_channel (bool, optional): 是否对每个通道独立进行下采样增强。默认为 True。
        p_per_channel (float, optional): 每个通道进行下采样增强的概率。默认为 1，表示每个通道都会进行下采样增强。
        channels (Optional[List[int]], optional): 需要进行下采样增强的通道索引列表。默认为 None，表示对所有通道进行下采样增强。
        order_downsample (int, optional): 下采样时使用的插值阶数。默认为 1。
        order_upsample (int, optional): 上采样时使用的插值阶数。默认为 0。
        ignore_axes (Optional[List[int]], optional): 忽略下采样的维度索引列表。默认为 None，表示对所有维度进行下采样。

    Returns:
        np.ndarray: 进行了线性下采样增强的数据样本，形状与输入数据样本相同。

    """
    if not isinstance(zoom_range, (list, tuple, np.ndarray)):
        zoom_range = [zoom_range]

    shp = np.array(data_sample.shape[1:])
    dim = len(shp)

    if not per_channel:
        if isinstance(zoom_range[0], (tuple, list, np.ndarray)):
            assert len(zoom_range) == dim
            zoom = np.array([np.random.uniform(i[0], i[1]) for i in zoom_range])
        else:
            zoom = np.random.uniform(zoom_range[0], zoom_range[1])

        target_shape = np.round(shp * zoom).astype(int)

        if ignore_axes is not None:
            for i in ignore_axes:
                target_shape[i] = shp[i]

    if channels is None:
        channels = list(range(data_sample.shape[0]))

    for c in channels:
        if np.random.uniform() < p_per_channel:
            if per_channel:
                if isinstance(zoom_range[0], (tuple, list, np.ndarray)):
                    assert len(zoom_range) == dim
                    zoom = np.array(
                        [np.random.uniform(i[0], i[1]) for i in zoom_range])
                else:
                    zoom = np.random.uniform(zoom_range[0], zoom_range[1])

                target_shape = np.round(shp * zoom).astype(int)
                if ignore_axes is not None:
                    for i in ignore_axes:
                        target_shape[i] = shp[i]

            downsampled = resize(
                data_sample[c].astype(float),
                target_shape,
                order=order_downsample,
                mode='edge',
                anti_aliasing=False)
            data_sample[c] = resize(
                downsampled,
                shp,
                order=order_upsample,
                mode='edge',
                anti_aliasing=False)

    return data_sample


def augment_gamma(data_sample,
                  gamma_range=(0.5, 2),
                  invert_image=False,
                  epsilon=1e-7,
                  per_channel=False,
                  retain_stats: Union[bool, Callable[[], bool]] = False):
    """
    对输入的数据样本进行伽马变换以增强对比度。

    Args:
        data_sample (np.ndarray): 输入的数据样本，形状为 (channels, height, width) 或 (height, width, channels)。
        gamma_range (Tuple[float, float], optional): 伽马值的取值范围，为包含两个元素的元组，表示伽马值的最小值和最大值。默认为 (0.5, 2)。
        invert_image (bool, optional): 是否对图像进行反转。默认为 False。
        epsilon (float, optional): 避免除以零的极小值。默认为 1e-7。
        per_channel (bool, optional): 是否对每个通道独立进行伽马变换。默认为 False。
        retain_stats (Union[bool, Callable[[], bool]], optional): 是否保持原始数据的均值和标准差不变。
            若为 True 或可调用对象，则保留；若为 False，则不保留。默认为 False。

    Returns:
        np.ndarray: 进行了伽马变换的数据样本，形状与输入数据样本相同。

    """
    if invert_image:
        data_sample = -data_sample

    if not per_channel:
        retain_stats_here = retain_stats() if callable(
            retain_stats) else retain_stats
        if retain_stats_here:
            mn = data_sample.mean()
            sd = data_sample.std()
        if np.random.random() < 0.5 and gamma_range[0] < 1:
            gamma = np.random.uniform(gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
        minm = data_sample.min()
        rnge = data_sample.max() - minm
        data_sample = np.power((
                (data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
        if retain_stats_here:
            data_sample = data_sample - data_sample.mean()
            data_sample = data_sample / (data_sample.std() + 1e-8) * sd
            data_sample = data_sample + mn
    else:
        for c in range(data_sample.shape[0]):
            retain_stats_here = retain_stats() if callable(
                retain_stats) else retain_stats
            if retain_stats_here:
                mn = data_sample[c].mean()
                sd = data_sample[c].std()
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(
                    max(gamma_range[0], 1), gamma_range[1])
            minm = data_sample[c].min()
            rnge = data_sample[c].max() - minm
            data_sample[c] = np.power((
                    (data_sample[c] - minm) / float(rnge + epsilon)),
                gamma) * float(rnge + epsilon) + minm
            if retain_stats_here:
                data_sample[c] = data_sample[c] - data_sample[c].mean()
                data_sample[c] = data_sample[c] / (
                        data_sample[c].std() + 1e-8) * sd
                data_sample[c] = data_sample[c] + mn
    if invert_image:
        data_sample = -data_sample
    return data_sample


def augment_mirroring(sample_data, sample_seg=None, axes=(0, 1, 2)):
    """
    对输入的数据样本进行镜像翻转增强。

    Args:
        sample_data (np.ndarray): 输入的数据样本，形状为 [channels, x, y] 或 [channels, x, y, z]。
        sample_seg (Optional[np.ndarray]): 对应的分割标签，形状与sample_data相同。默认为None。
        axes (Tuple[int, ...]): 需要进行镜像翻转的维度索引。默认为(0, 1, 2)，表示在x, y, z三个维度上都可能进行镜像翻转。

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: 镜像翻转后的数据样本和对应的分割标签。若未传入sample_seg，则只返回镜像翻转后的数据样本。

    Raises:
        ValueError: 如果sample_data和sample_seg的维度不是3或4，则抛出ValueError异常。
    """
    if (len(sample_data.shape) != 3) and (len(sample_data.shape) != 4):
        raise ValueError(
            "Invalid dimension for sample_data and sample_seg. sample_data and sample_seg should be either "
            "[channels, x, y] or [channels, x, y, z]")
    if 0 in axes and np.random.uniform() < 0.5:
        sample_data[:, :] = sample_data[:, ::-1]
        if sample_seg is not None:
            sample_seg[:, :] = sample_seg[:, ::-1]
    if 1 in axes and np.random.uniform() < 0.5:
        sample_data[:, :, :] = sample_data[:, :, ::-1]
        if sample_seg is not None:
            sample_seg[:, :, :] = sample_seg[:, :, ::-1]
    if 2 in axes and len(sample_data.shape) == 4:
        if np.random.uniform() < 0.5:
            sample_data[:, :, :, :] = sample_data[:, :, :, ::-1]
            if sample_seg is not None:
                sample_seg[:, :, :, :] = sample_seg[:, :, :, ::-1]
    return sample_data, sample_seg
