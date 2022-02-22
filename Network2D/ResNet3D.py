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

    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=400, use_nl=False):
        self.inplanes = 64
        super(I3Res50, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(2, 2, 2), padding=(2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 0, 0))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        nonlocal_mod = 2 if use_nl else 1000
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, temp_conv=[1, 1, 1], temp_stride=[1, 1, 1])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, temp_conv=[1, 0, 1, 0],
                                       temp_stride=[1, 1, 1, 1], nonlocal_mod=nonlocal_mod)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, temp_conv=[1, 0, 1, 0, 1, 0],
                                       temp_stride=[1, 1, 1, 1, 1, 1], nonlocal_mod=nonlocal_mod)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, temp_conv=[0, 1, 0], temp_stride=[1, 1, 1])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.drop = nn.Dropout(0.5)

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
        feature_map, dis_map = inputs[0], inputs[1]
        x = self.conv1(feature_map)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x, _ = self.layer1([x, dis_map])
        x = self.maxpool2(x)
        x, _ = self.layer2([x, dis_map])
        x, _ = self.layer3([x, dis_map])
        x, _ = self.layer4([x, dis_map])

        x = self.avgpool(x)
        x = self.drop(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
    #
    # def forward_multi(self, x):
    #     clip_preds = []
    #     for clip_idx in range(x.shape[1]):  # B, 10, 3, 3, 32, 224, 224
    #         spatial_crops = []
    #         for crop_idx in range(x.shape[2]):
    #             clip = x[:, clip_idx, crop_idx]
    #             clip = self.forward_single(clip)
    #             spatial_crops.append(clip)
    #         spatial_crops = torch.stack(spatial_crops, 1).mean(1)  # (B, 400)
    #         clip_preds.append(spatial_crops)
    #     clip_preds = torch.stack(clip_preds, 1).mean(1)  # (B, 400)
    #     return clip_preds
    #
    # def forward(self, batch):
    #
    #     # 5D tensor == single clip
    #     if batch['frames'].dim() == 5:
    #         pred = self.forward_single(batch['frames'])
    #
    #     # 7D tensor == 3 crops/10 clips
    #     elif batch['frames'].dim() == 7:
    #         pred = self.forward_multi(batch['frames'])
    #
    #     loss_dict = {}
    #     if 'label' in batch:
    #         loss = F.cross_entropy(pred, batch['label'], reduction='none')
    #         loss_dict = {'loss': loss}
    #
    #     return pred, loss_dict


# -----------------------------------------------------------------------------------------------#

def i3_res50(num_classes):
    net = I3Res50(num_classes=num_classes, use_nl=False)
    # state_dict = torch.load('pretrained/i3d_r50_kinetics.pth')
    # net.load_state_dict(state_dict)
    # freeze_bn(net, "net") # Only needed for finetuning. For validation, .eval() works.
    return net


if __name__ == '__main__':
    net = i3_res50(2)
    inputs = torch.randn(1, 3, 100, 100, 50)
    dis_map = torch.randn(1, 1, 100, 100, 50)
    prediction = net([inputs, dis_map])
    print(prediction.shape)
    print(net)
    # inp = {'frames': torch.rand(4, 3, 32, 224, 224)}
    # pred, losses = net(inp)
