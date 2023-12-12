import torch
import torch.nn as nn
from layer_blocks import create_3x3_convolution as conv3x3
from layer_blocks import apply_layers_and_capture_outputs


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, input_channels, output_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(input_channels, output_channels, stride)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(output_channels, output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = [block(self.in_channels, planes, stride, downsample)]
        self.in_channels = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        res1 = apply_layers_and_capture_outputs(x, self.layer1)
        res2 = apply_layers_and_capture_outputs(res1[1], self.layer2)
        res3 = apply_layers_and_capture_outputs(res2[1], self.layer3)
        res4 = apply_layers_and_capture_outputs(res3[1], self.layer4)

        return x, res1, res2, res3, res4


def load_resnet_rgb_model(pretrained=True, model_path=None):
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    
    if pretrained:
        if model_path is None:
            # Directly load the pretrained model state dict
            pretrained_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet18-5c106cde.pth')
        else:
            # Load state dict from a provided file path
            pretrained_dict = torch.load(model_path, map_location='cpu')

        model_dict = model.state_dict()
        
        # Filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model

