"""
    SINet for image segmentation, implemented in PyTorch.
    Original paper: 'SINet: Extreme Lightweight Portrait Segmentation Networks with Spatial Squeeze Modules and
    Information Blocking Decoder,' https://arxiv.org/abs/1911.09099.
"""


import os
import torch
import torch.nn as nn

from pytorchcv.models.sinet import SINet


def get_sinet():
    """
    Create SINet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    kernel_sizes_list = [
        [[3, 5], [3, 3], [3, 3]],
        [[3, 5], [3, 3], [5, 5], [3, 5], [3, 5], [3, 5], [3, 3], [5, 5], [3, 5], [3, 5]]]
    scale_factors_list = [
        [[1, 1], [0, 1], [0, 1]],
        [[1, 1], [0, 1], [1, 4], [2, 8], [1, 1], [1, 1], [0, 1], [1, 8], [2, 4], [0, 2]]]

    chnn = 4
    dims = [24] + [24 * (i + 2) + 4 * (chnn - 1) for i in range(3)]

    dim1 = dims[0]
    dim2 = dims[1]
    dim3 = dims[2]
    dim4 = dims[3]

    p = len(kernel_sizes_list[0])
    q = len(kernel_sizes_list[1])

    channels_list = [[dim2] * p, ([dim3] * (q // 2)) + ([dim4] * (q - q // 2))]
    use_residual_list = [[0] + ([1] * (p - 1)), [0] + ([1] * (q // 2 - 1)) + [0] + ([1] * (q - q // 2 - 1))]

    down_channels_list = [dim1, dim2]

    net = SINet(
        down_channels_list=down_channels_list,
        channels_list=channels_list,
        kernel_sizes_list=kernel_sizes_list,
        scale_factors_list=scale_factors_list,
        use_residual_list=use_residual_list,
        dim2=dims[1],
        bn_eps=1e-3,
        in_size=(128,128),
        num_classes = 3)

    return net
