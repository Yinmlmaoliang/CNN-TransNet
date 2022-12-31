import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = x * self.sigmoid(out)
        return out


class MMFPv1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MMFPv1, self).__init__()
        self.channel_att = ChannelAttention(in_planes=in_channels)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                               stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.flatten(2)
        x = x.permute(0, 2, 1)
        return x


class MMFPv2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MMFPv2, self).__init__()
        self.channel_att = ChannelAttention(in_planes=in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7,
                               stride=7, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.flatten(2)
        x = x.permute(0, 2, 1)
        return x
