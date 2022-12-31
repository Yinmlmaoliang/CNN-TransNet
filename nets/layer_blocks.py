import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=stride, bias=False)


def extract_res(x, layers):
    res = []
    for idx, layer in enumerate(layers):
        x = layer(x)
        res.append(x)
    return res
