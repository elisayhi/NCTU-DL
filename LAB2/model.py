import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from dataloader import read_bci_data

class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
                nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                )
        self.depthwiseConv = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ELU(alpha=1.0),
                nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
                nn.Dropout(p=0.25),
                )
        self.separableConv = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ELU(alpha=1.0),
                nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
                nn.Dropout(p=0.4),
                )
        self.classify = nn.Sequential(
                nn.Linear(in_features=736, out_features=2,bias=True),
                )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(x.size(0), -1)   # flatten
        x = self.classify(x)
        return x
