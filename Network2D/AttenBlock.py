import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data.sampler as sampler
from torch.autograd import Variable

'''
https://github.com/superxuang/amta-net/blob/master/model.py
'''


class att_module(nn.Module):
    expansion = 4
    def __init__(self, in1_ch, in2_ch, out_ch, maxpool):
        super(att_module, self).__init__()
        in1_ch, in2_ch, out_ch = in1_ch * self.expansion, in2_ch * self.expansion, out_ch * self.expansion

        self.att_conv = nn.Sequential(
            nn.Conv2d(in_channels=in1_ch + in2_ch, out_channels=in1_ch + in2_ch, kernel_size=(1, 1), padding=0),
            nn.BatchNorm2d(in1_ch + in2_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in1_ch + in2_ch, out_channels=in1_ch + in2_ch, kernel_size=(1, 1), padding=0),
            nn.BatchNorm2d(in1_ch + in2_ch),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in1_ch + in2_ch, out_channels=out_ch, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(inplace=True)
        )
        self.maxpool = maxpool
        if self.maxpool:
            self.resample = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.resample = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3, 3), stride=(2, 2), padding=1)

    def forward(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        y = torch.cat([x1, x2], dim=1)
        att_mask = self.att_conv(y)
        y = att_mask * y
        y = self.conv(y)
        y = self.resample(y)
        return y


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, input_channels=3, num_classes=1000, maxpool=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        ################################################################################################################
        self.atten_conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.atten_bn1 = nn.BatchNorm2d(64)
        self.atten_relu = nn.ReLU(inplace=True)
        self.atten_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.atten_layer0 = att_module(int(64/4), int(64/4), 64, maxpool)
        self.atten_layer1 = att_module(64, 64, 64, maxpool)
        self.atten_layer2 = att_module(128, 64, 128, maxpool)
        self.atten_layer3 = att_module(256, 128, 256, maxpool)
        self.atten_layer4 = att_module(512, 256, 512, maxpool)

        ################################################################################################################
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(nn.Linear(512 * block.expansion * 2,  int(512 * block.expansion/2)),
                                 nn.Dropout(0.5),
                                 nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(int(512 * block.expansion/2), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        y = self.atten_conv1(y)
        y = self.atten_bn1(y)
        y = self.atten_relu(y)

        y = self.atten_layer0(x, y)

        x = self.maxpool(x)

        x = self.layer1(x)
        y = self.atten_layer1(x, y)
        x = self.layer2(x)
        y = self.atten_layer2(x, y)
        x = self.layer3(x)
        y = self.atten_layer3(x, y)
        x = self.layer4(x)
        y = self.atten_layer4(x, y)

        x = self.avgpool(x)
        y = self.avgpool(y)
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)

        x = self.fc2(self.fc1(torch.cat([x, y], dim=1)))

        return x


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


if __name__ == '__main__':
    model = resnet50(input_channels=9, num_classes=4, )
    input_image = torch.from_numpy(np.random.rand(3, 9, 200, 200)).float()
    input_mask = torch.from_numpy(np.random.rand(3, 9, 200, 200)).float()
    pred = model(input_image, input_mask)
    print(pred.shape)