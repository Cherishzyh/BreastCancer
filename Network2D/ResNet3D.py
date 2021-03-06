import torch
import torch.nn as nn
import torch.nn.functional as F
import math

''' https://github.com/Tushar-N/pytorch-resnet3d/blob/master/models/resnet.py '''



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out

        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride, downsample, temp_conv, temp_stride, use_nl=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1 + temp_conv * 2, 1, 1), stride=(temp_stride, 1, 1),
                               padding=(temp_conv, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1),
                               bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.ca = ChannelAttention(planes * 4)
        self.sa_fm = SpatialAttention()

        self.conv1x1 = nn.Conv3d(planes * 8, planes * 4, kernel_size=(1, 1, 1))

        outplanes = planes * 4

    def forward(self, inputs):
        feature_map, dis_map = inputs[0], inputs[1]
        residual = feature_map

        out = self.conv1(feature_map)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        shape = out.shape[2:]
        dis_map_resize = F.interpolate(dis_map, size=shape, mode='trilinear', align_corners=True)

        out = self.ca(out) * out
        out_fm = self.sa_fm(out) * out
        out_dm = dis_map_resize * out

        out = self.conv1x1(torch.cat([out_fm, out_dm], dim=1))

        if self.downsample is not None:
            residual = self.downsample(feature_map)

        out += residual
        out = self.relu(out)

        return [out, dis_map]



class I3Res50(nn.Module):

    def __init__(self, input_planes=3, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=400, use_nl=False):
        self.inplanes = 64
        super(I3Res50, self).__init__()
        self.conv1 = nn.Conv3d(input_planes, 64, kernel_size=(5, 7, 7), stride=(2, 2, 2), padding=(2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 0, 0))
        # self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        nonlocal_mod = 2 if use_nl else 1000
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, temp_conv=[1, 1, 1], temp_stride=[1, 1, 1])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, temp_conv=[1, 0, 1, 0],
                                       temp_stride=[1, 1, 1, 1], nonlocal_mod=nonlocal_mod)
        self.conv2 = nn.Conv3d(128 * block.expansion, 128 * block.expansion, kernel_size=(3, 1, 1), stride=(2, 1, 1),
                               padding=(1, 0, 0))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, temp_conv=[1, 0, 1, 0, 1, 0],
                                       temp_stride=[1, 1, 1, 1, 1, 1], nonlocal_mod=nonlocal_mod)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, temp_conv=[0, 1, 0], temp_stride=[1, 1, 1])
        self.conv3 = nn.Conv3d(512 * block.expansion, 512 * block.expansion, kernel_size=(3, 1, 1), stride=(2, 1, 1),
                               padding=(1, 0, 0))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Sequential(nn.Linear(512 * block.expansion, 128 * block.expansion),
                                 nn.Dropout(0.5),
                                 nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(128 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride, temp_conv, temp_stride, nonlocal_mod=1000):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or temp_stride[0] != 1:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=(1, 1, 1),
                          stride=(temp_stride[0], stride, stride), padding=(0, 0, 0), bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, temp_conv[0], temp_stride[0], False))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, temp_conv[i], temp_stride[i],
                                i % nonlocal_mod == nonlocal_mod - 1))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        feature_map, dis_map = inputs[0], inputs[1]   # 1, 3, 50, 100, 100
        # print(feature_map.shape)
        x = self.conv1(feature_map)                   # 1, 64, 25, 50, 50
        # print(x.shape)
        x = self.bn1(x)
        x = self.relu(x)

        x, _ = self.layer1([x, dis_map])              # 1, 256, 25, 50, 50
        # print(x.shape)
        x, _ = self.layer2([x, dis_map])              # 1, 512, 25, 25, 25
        # print(x.shape)
        x = self.conv2(x)                             # 1, 512, 13, 25, 25
        # print(x.shape)
        x, _ = self.layer3([x, dis_map])              # 1, 1024, 13, 13, 13
        # print(x.shape)
        x, _ = self.layer4([x, dis_map])              # 1, 2048, 13, 7, 7
        # print(x.shape)
        x = self.conv3(x)                             # 1, 2048, 7, 7, 7
        # print(x.shape)

        x = self.avgpool(x)                           # 1, 2048, 1, 1, 1
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)                               # 1, 512
        # print(x.shape)
        x = self.fc2(x)

        return x


# -----------------------------------------------------------------------------------------------#

def i3_res50(input_planes, num_classes):
    net = I3Res50(input_planes=input_planes, num_classes=num_classes, use_nl=False)
    return net


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net = i3_res50(2)
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), 'gpus')
        net = nn.DataParallel(net)

    net.to(device)
    inputs = torch.randn(12, 3, 50, 100, 100).to(device)
    dis_map = torch.randn(12, 1, 50, 100, 100).to(device)
    prediction = net([inputs, dis_map])
    print(prediction.shape)
    # print(net)
    # inp = {'frames': torch.rand(4, 3, 32, 224, 224)}
    # pred, losses = net(inp)
