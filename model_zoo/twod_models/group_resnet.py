
from functools import partial
from inspect import signature

import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from .common import TemporalPooling
from .temporal_modeling import temporal_modeling_module
from model_zoo.inflate_from_2d_model import convert_rgb_model_to_group

__all__ = ['group_resnet']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3_group(in_planes, out_planes, groups=1, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, groups=groups, stride=stride,
                     padding=1, bias=False)

def conv1x1_group(in_planes, out_planes, groups=1, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, groups=groups, stride=stride, bias=False)

'''
class TemporalChannelExpansion(nn.Module):
    def __init__(self, inplane, groups=1, expansion=1):
        super(TemporalChannelExpansion, self).__init__()
        self.expansion = expansion
        self.groups = groups
        self.conv = conv3x3(inplane//groups * 3, int(inplane//groups * expansion), kernel_size=3, bias=False)

    def forward(self, x):
        if self.expansion == 1:
            return x

        # input: NXCXWXH
        n, c, w, h = x.shape

        # each frame owns C/groups channels
        new_c = c // self.groups

        out =
        out = self.conv(out)
        out = out[]

        return out
'''

class BasicBlockGroup(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, num_frames, groups=1, stride=1, downsample=None, temporal_module=None):
        super(BasicBlockGroup, self).__init__()
#        self.tc_squeeze = conv1x1_group(inplanes, planes, groups)
#        self.tc_expand = TemporalChannelExpansion(inplanes, groups, temporal_expansion)
        self.conv1 = conv3x3_group(inplanes, planes, groups, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_group(planes, planes, groups)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.tam = temporal_module(duration=num_frames, channels=inplanes) \
            if temporal_module is not None else None

    def forward(self, x):
        identity = x
        if self.tam is not None:
            x = self.tam(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            if self.tam is not None:
                if 'GroupTSM' in self.tam.name():
                    identity = self.downsample(identity)
                else:
                    identity = self.downsample(x)
            else:
                identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckGroup(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, num_frames, groups=1, stride=1, downsample=None, temporal_module=None):
        super(BottleneckGroup, self).__init__()

        self.conv1 = conv1x1_group(inplanes, planes, groups)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3_group(planes, planes, groups, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1_group(planes, planes * self.expansion, groups)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.tam = temporal_module(duration=num_frames, channels=inplanes) \
            if temporal_module is not None else None

    def forward(self, x):
        identity = x
        if self.tam is not None:
           x = self.tam(x)

        # squeeze here
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            if self.tam is not None:
                if 'GroupTSM' in self.tam.name():
                    identity = self.downsample(identity)
                else:
                    identity = self.downsample(x)
            else:
                identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        # expansion here

        return out


class GroupResNet(nn.Module):

    def __init__(self, depth, num_frames, num_classes=1000, dropout=0.5, zero_init_residual=False,
                 without_t_stride=False, temporal_module=None, fpn_dim=-1, pooling_method='max',
                 input_channels=3):
        super(GroupResNet, self).__init__()

        self.pooling_method = pooling_method.lower()
        block = BasicBlockGroup if depth < 50 else BottleneckGroup
        layers = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]}[depth]

        self.depth = depth
        self.temporal_module = temporal_module
        self.num_frames = num_frames
        self.orig_num_frames = num_frames
        self.num_classes = num_classes
        self.without_t_stride = without_t_stride
        self.fpn_dim = fpn_dim

        self.inplanes = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.downsample1 = nn.Sequential(conv1x1_group(64, 64//self.orig_num_frames),
                nn.BatchNorm2d(64 // self.orig_num_frames),)

        self.layer1 = self._make_layer(block, 64, layers[0])
        if not self.without_t_stride:
            self.pool1 = TemporalPooling(self.num_frames, 3, 2, self.pooling_method)
            self.num_frames = self.num_frames // 2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        if not self.without_t_stride:
            self.pool2 = TemporalPooling(self.num_frames, 3, 2, self.pooling_method)
            self.num_frames = self.num_frames // 2
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        if not self.without_t_stride:
            self.pool3 = TemporalPooling(self.num_frames, 3, 2, self.pooling_method)
            self.num_frames = self.num_frames // 2
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)

        if self.fpn_dim > 0:
            self.fpn = FPN(self.fpn_dim)
            self.fc2 = nn.Linear(fpn_dim, num_classes)
            self.fc3 = nn.Linear(fpn_dim, num_classes)
            self.fc4 = nn.Linear(fpn_dim, num_classes)
            self.fc5 = nn.Linear(fpn_dim, num_classes)
        else:
            self.fc = nn.Linear(512 * block.expansion // self.orig_num_frames, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckGroup):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlockGroup):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1_group(self.inplanes, planes * block.expansion, groups=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.orig_num_frames, self.num_frames, stride, downsample,
                            temporal_module=self.temporal_module))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.num_frames, groups=self.orig_num_frames, 
                                temporal_module=self.temporal_module))

        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size, c_t, h, w = x.shape
        x = x.view(batch_size * self.orig_num_frames, c_t // self.orig_num_frames, h, w)
        #N*FxCxWxH
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # downsample to fit group size
        x = self.downsample1(x)
        x = self.relu(x)
        #NxF*CXWXH
        x = x.view((batch_size, -1)+x.shape[-2:])
        fp1 = self.maxpool(x)

        fp2 = self.layer1(fp1)
        fp2_d = self.pool1(fp2) if not self.without_t_stride else fp2
        fp3 = self.layer2(fp2_d)
        fp3_d = self.pool2(fp3) if not self.without_t_stride else fp3
        fp4 = self.layer3(fp3_d)
        fp4_d = self.pool3(fp4) if not self.without_t_stride else fp4
        fp5 = self.layer4(fp4_d)

        if self.fpn_dim > 0:
            fp2, fp3, fp4, fp5 = self.fpn(fp2, fp3, fp4, fp5, batch_size)
            pred2 = torch.mean(self.fc2(fp2).view(batch_size, -1, self.num_classes),
                               dim=1, keepdim=True)
            pred3 = torch.mean(self.fc3(fp3).view(batch_size, -1, self.num_classes),
                               dim=1, keepdim=True)
            pred4 = torch.mean(self.fc4(fp4).view(batch_size, -1, self.num_classes),
                               dim=1, keepdim=True)
            pred5 = torch.mean(self.fc5(fp5).view(batch_size, -1, self.num_classes),
                               dim=1, keepdim=True)

            out = torch.cat((pred2, pred3, pred4, pred5), dim=1)
        else:
            x = self.avgpool(fp5)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = x.view(batch_size * self.orig_num_frames, -1)
            x = self.fc(x)

            n_t, c = x.shape
            out = x.view(batch_size, -1, c)

        # average the prediction from all frames
        out = torch.mean(out, dim=1)

        return out

    def mean(self, modality='rgb'):
        return [0.485, 0.456, 0.406] if modality == 'rgb' else [0.5]

    def std(self, modality='rgb'):
        return [0.229, 0.224, 0.225] if modality == 'rgb' else [np.mean([0.229, 0.224, 0.225])]

    @property
    def network_name(self):
        name = ''
        if self.temporal_module is not None:
            param = signature(self.temporal_module).parameters
            temporal_module = str(param['name']).split("=")[-1][1:-1]
            blending_frames = str(param['blending_frames']).split("=")[-1]
            blending_method = str(param['blending_method']).split("=")[-1][1:-1]
            dw_conv = True if str(param['dw_conv']).split("=")[-1] == 'True' else False
            name += "{}-b{}-{}{}-".format(temporal_module, blending_frames,
                                         blending_method,
                                         "" if dw_conv else "-allc")
        name += 'group-resnet-{}'.format(self.depth)
        if not self.without_t_stride:
            name += "-ts-{}".format(self.pooling_method)
        if self.fpn_dim > 0:
            name += "-fpn{}".format(self.fpn_dim)

        return name


def group_resnet(depth, num_classes, without_t_stride, groups, temporal_module_name,
           dw_conv, blending_frames, blending_method, dropout, fpn_dim, pooling_method,
           input_channels, imagenet_pretrained=True, **kwargs):

    temporal_module = partial(temporal_modeling_module, name=temporal_module_name,
                              dw_conv=dw_conv,
                              blending_frames=blending_frames,
                              blending_method=blending_method) if temporal_module_name is not None \
        else None

    model = GroupResNet(depth, num_frames=groups, num_classes=num_classes,
                   without_t_stride=without_t_stride,
                   temporal_module=temporal_module, dropout=dropout, fpn_dim=fpn_dim,
                   pooling_method=pooling_method, input_channels=input_channels)

#    for name, param in model.named_parameters():
#        print (name, param.data.shape)


    if imagenet_pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet{}'.format(depth)], map_location='cpu')
        state_dict.pop('fc.weight', None)
        state_dict.pop('fc.bias', None)
        #state_dict = convert_rgb_model_to_group(state_dict, groups)
        #model.load_state_dict(state_dict, strict=False)

    return model


if __name__ == '__main__':
    from torchsummary import torchsummary
    model = group_resnet(50, 187, without_t_stride=False, groups=8, dropout=0.5,
                   temporal_module_name='TAM', dw_conv=True, blending_frames=3,
                   blending_method='sum', fpn_dim=-1)

    dummy_data = (24, 224, 224)
    model.eval()
    model_summary = torchsummary.summary(model, input_size=dummy_data)
    print(model.network_name)

