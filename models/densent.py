# -*- coding: utf-8 -*-

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import numpy as np
import time
import copy

class Bottleneck(nn.Module): #DenseNet-B
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        inner_channels = 4 * growth_rate

        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, growth_rate, 3, stride=1, padding=1, bias=False)
        )
        
    def forward(self, x):
        out = torch.cat((self.residual(x), x), 1)
        # print('B', out.shape)
        return out

class SingleLayer(nn.Module):
    def __init__(self, in_channels, growthRate):
        super(SingleLayer, self).__init__()
        
        self.single = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growthRate, 3, padding=1, bias=False)
        )
    
    def forward(self, x):
        out = torch.cat((self.single(x), x), 1)
        # print(out.shape)
        return out

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(2, stride=2)
        )
    
    def forward(self, x):
        out = self.transition(x)
        # print('T', out.shape)
        return out

class DenseNet(nn.Module):
    def __init__(self, growthRate, block_config, depth, reduction, nClasses, bottleneck):
        super(DenseNet, self).__init__()

        # nDenseBlocks = (depth - 4) // 3
        # if Bottleneck:
        #     nDenseBlocks  //= 2

        nChannels = 2 * growthRate
        self.initial = nn.Sequential(
            nn.Conv2d(3, nChannels, 3, padding=1, bias=False),
            nn.BatchNorm2d(nChannels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.features = nn.Sequential()
        for i, num_layers in enumerate(block_config):
            self.features.add_module('denseblock%d' % (i+1), 
                                     self.dense_layer(nChannels, growthRate, num_layers, bottleneck))
            nChannels += num_layers * growthRate
            nOutChannels = int(np.floor(nChannels * reduction))
            if i != len(block_config) - 1:
                self.features.add_module('transition%d' % (i+1),
                                     Transition(nChannels, nOutChannels))
                nChannels = nOutChannels

        self.features.add_module('FinalNorm', nn.BatchNorm2d(nChannels))
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(nChannels, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def dense_layer(self, nChannels, growthRate, num_layers, bottleneck):
        layers = []
        for i in range(num_layers):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.initial(x)
        # print(out.shape)
        out = self.features(out)
        # print(out.shape)
        out = self.avg_pool(out)
        # print(out.shape )
        out = self.classifier(out.view(out.size(0), -1))
        return out



if __name__ == '__main__':
    import torchsummary
    model = DenseNet(12, (6, 12, 24, 6), 32, 0.5, 10, True)
    torchsummary.summary(model, (3, 224, 224))
    
    x = torch.randn(3, 3, 64, 64)
    output = model(x)
    print(output.shape)

