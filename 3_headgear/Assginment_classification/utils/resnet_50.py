# './utils/resnet_50.py'
import torch
import torch.nn as nn

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_rate=0.0):
        super(Bottleneck, self).__init__()
        
        # TODO: Define the layers
        self.conv1 = conv1x1(inplanes, planes)
        # self.bn1 = # fill this in # BatchNorm2d
        self.conv2 = conv3x3(planes, planes, stride)
        # self.bn2 = # fill this in # BatchNorm2d
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        # self.dropout = # fill this in # Dropout
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # TODO: Define the forward pass
        # out = self.'# fill this in' # conv1
        # out = self.'# fill this in' # bn1
        # out = self.'# fill this in' # relu
        out = self.dropout(out)

        # out = self.'# fill this in'
        # out = self.'# fill this in'
        # out = self.'# fill this in'
        out = self.dropout(out)

        # out = self.'# fill this in'
        # out = self.'# fill this in'
        out = self.dropout(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        out = self.dropout(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, dropout_rate=0.0):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.dropout_rate = dropout_rate
        
        # TODO: Define the layers referring to the ResNet
        # self.conv1 = # fill this in(#, #, kernel_size, stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = # fill this in(kernel_size, stride, padding=1)
        
        # TODO: Follow the ResNet architecture and fill in the blanks
        # self.layer1 = self._make_layer(block, # fill this in , layers[0])
        # self.layer2 = self._make_layer(block, # fill this in, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, # fill this in, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, # fill this in, layers[3], stride=2)
        # self.avgpool = # fill this in # use AdaptiveAvgPool2d
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.dropout_rate))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout_rate=self.dropout_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        # TODO: Define the forward pass
        # x = self.'# fill this in'
        # x = self.'# fill this in'
        # x = self.# fill this in'
        x = self.dropout(x)
        # x = self. # maxpooling

        # TODO: Define the Layer 1 ~ 4
        # x = self.'# fill this in'
        # x = self.'# fill this in'
        # x = self.'# fill this in'
        # x = self.'# fill this in'

        # x = self.'# fill this in # avreage pool
        # x = torch.flatten(# fill this in) # flatten
        x = self.dropout(x)
        # x = self.'# fill this in' # fully connected layer

        return x

def resnet50(num_classes, dropout_rate=0.0):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, dropout_rate)
