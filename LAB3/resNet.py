import numpy
import torch
from torch import nn

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_plane, out_plane, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(in_plane, out_plane, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_plane)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_plane, out_plane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(out_plane)
        self.downsample = downsample
        self.plane = [in_plane, out_plane]

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        #print(f'in plane: {self.plane[0]} out plane: {self.plane[1]}')

        if self.downsample != None:
            residual = self.downsample(residual) # if in_plane != out_plane, the need to up or downsample the residual

        #print(f'x: {x.size()} out: {out.size()} residual: {residual.size()}')
        #print('downsample', self.downsample)
        #print('residual', residual)
        #print('out', out)
        out += residual
        out = self.relu(out)

        return out

class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_plane, out_plane, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_plane, out_plane, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_plane)
        self.conv2 = nn.Conv2d(out_plane, out_plane, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_plane)
        self.conv3 = nn.Conv2d(out_plane, out_plane*BottleNeck.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_plane*BottleNeck.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

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

        if self.downsample != None:
            residual = self.downsample(residual)

        #print(f'x: {x.size()} out: {out.size()} residual: {residual.size()}')
        #print('downsample', self.downsample)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, conv2_inwidth=64, norm_layer=None):
        """
        layers: numbers of each block
        """
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        planes = [int(conv2_inwidth*2**i) for i in range(4)]
        self.inchannel = planes[0]

        self.conv1 = nn.Sequential(
                    nn.Conv2d(3, planes[0], kernel_size=7, stride=2, padding=3, bias=False),
                    norm_layer(planes[0]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )

        self.layer1 = self._make_layer(block, planes[0], layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(planes[3]*block.expansion, 5)

    def _make_layer(self, block, planes, blocks, stride=1, groups=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        outchannel = planes*block.expansion
        downsample = None
        if stride != 1 or self.inchannel != planes*block.expansion:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                    norm_layer(planes*block.expansion),
                    )
        layers = []
        layers.append(block(self.inchannel, planes, stride, downsample, norm_layer))
        self.inchannel = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inchannel, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(x.size(0), -1)
        out = self.fc(out)

        return out 

        
        
