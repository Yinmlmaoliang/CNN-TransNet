import torch
import torch.nn as nn
from layer_blocks import conv3x3
from layer_blocks import extract_res
import numpy as np


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # Conv1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # Conv2_x
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        # Conv3_x
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # Conv4_x
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # Conv5_x
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        conv1 = self.relu(x)
        x = self.maxpool(conv1)
        res1 = extract_res(x, self.layer1)
        res2 = extract_res(res1[1], self.layer2)
        res3 = extract_res(res2[1], self.layer3)
        res4 = extract_res(res3[1], self.layer4)

        return conv1, res1, res2, res3, res4


def extraction_rgb(pretrained=True):
    # trained(bool): if True, returns a model pre-trained om ImageNet
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    model_path = 'resnet18-5c106cde.pth'
    if pretrained:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # print("\nSuccessful Load Key:", str(load_key), "……\nSuccessful Load Key Num:", len(load_key))
        # print("\nFail To Load Key:", str(no_load_key), "……\nFail To Load Key num:", len(no_load_key))

    return model
