import torch
import torch.nn as nn
import torch.nn.functional as F

class Aspp(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Aspp, self).__init__()
        channel = int(inchannel/4)
        self.Conv1 = nn.Conv2d(inchannel, channel, 1)
        self.bn1 = nn.BatchNorm2d(channel)
        self.Conv3r6 = nn.Conv2d(inchannel, channel, 3, stride=1, padding=6, dilation=6)
        self.bn3r6 = nn.BatchNorm2d(channel)
        self.Conv3r12 = nn.Conv2d(inchannel, channel, 3, stride=1, padding=12, dilation=12)
        self.bn3r12 = nn.BatchNorm2d(channel)
        self.Conv3r18 = nn.Conv2d(inchannel, channel, 3, stride=1, padding=18, dilation=18)
        self.bn3r18 = nn.BatchNorm2d(channel)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.conv_pool = nn.Conv2d(inchannel, channel, 1)
        self.bn_conv_pool = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.Conv1f = nn.Conv2d(5*channel, outchannel, 1)

    def forward(self, x):
        F_h = x.size()[2]
        F_w = x.size()[3]
        x1 = self.relu(self.bn1(self.Conv1(x)))
        x2 = self.relu(self.bn3r6(self.Conv3r6(x)))
        x3 = self.relu(self.bn3r12(self.Conv3r12(x)))
        x4 = self.relu(self.bn3r18(self.Conv3r18(x)))
        x5 = self.relu(self.bn_conv_pool(self.conv_pool(self.pool(x))))
        x5 = F.upsample(x5, size=(F_h, F_w), mode='bilinear')
        x = torch.cat([x1,x2,x3,x4,x5], 1)
        x = self.relu(x)
        x = self.Conv1f(x)
        return x
