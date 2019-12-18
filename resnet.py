import torch
import torch.nn as nn
from collections import OrderedDict

model_dict = {
    'resnet101':[3,4,23,3]
}

class BasicNeck(nn.Module):
    def __init__(self,in_plance,plance,stride=1, reduce_channel = None):
        super(BasicNeck, self).__init__()
        self.bn = nn.BatchNorm2d()
        self.relu = nn.ReLU()

        self.Conv3 = nn.Conv2d(plance, plance, 3, stride=1)
        self.reduce_channel = reduce_channel

    def forward(self, x):
        input = x
        x = self.relu(self.bn(self.Conv3(x)))
        x = self.bn(self.Conv3(x))
        if self.reduce_channel is not None:
            input = self.reduce_channel(input)
        return self.relu(x+input)


class BottleNeck(nn.Module):
    def __init__(self,in_plance,plance,stride=1, reduce_channel = None):
        super(BottleNeck, self).__init__()

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(plance)
        self.Conv1_1 = nn.Conv2d(in_plance, plance, 1, stride=1)
        self.bn2 = nn.BatchNorm2d(plance)
        self.Conv3 = nn.Conv2d(plance, plance, 3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(plance*4)
        self.Conv1_2 = nn.Conv2d(plance, plance*4, 1, stride=1)
        self.reduce_channel = reduce_channel

    def forward(self, x):
        input = x
        x = self.relu(self.bn1(self.Conv1_1(x)))
        x = self.relu(self.bn2(self.Conv3(x)))
        x = self.bn3(self.Conv1_2(x))
        if self.reduce_channel is not None:
            input = self.reduce_channel(input)
        return self.relu(x+input)



class Resnet(nn.Module):
    def __init__(self, name, in_channel=3, num_class = 1000):
        super(Resnet, self).__init__()
        block_nums = model_dict[name]
        blockname = BottleNeck if int(name[6:])>34 else BasicNeck

        layers = OrderedDict()
        layers['Conv'] = nn.Conv2d(in_channel, 64, 7, stride=2, padding=3)
        layers['Pool'] = nn.MaxPool2d(3, stride=2, padding=1)

        self.layers1 = nn.Sequential(layers)
        self.layers2 = Block(blockname, block_nums[0], 64, 64, stride = 1, padding = 1)
        self.layers3 = Block(blockname, block_nums[1],256, 128, stride = 2, padding = 1)
        self.layers4 = Block(blockname, block_nums[2],512, 256, stride = 2, padding = 1)
        self.layers5 = Block(blockname, block_nums[3],1024, 512, stride = 2, padding = 1)
        self.outchannel5 = 2048
        self.outchannel4 = 1024
        self.outchannel3 = 512
        self.outchannel2 = 256
        self.outchannel1 = 64
        self.GAP = nn.AdaptiveAvgPool2d(7)
        self.dp = nn.Dropout()
        self.fc = nn.Linear(2048, num_class)

    def forward(self, x):
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)
        x = self.layers5(x)
        x = self.GAP(x)
        x.squeeze_()
        x = self.dp(x)
        x = self.fc(x)
        return x


class Block(nn.Module):
    def __init__(self, blockname, num, inplance, plance, stride = 1, padding = 1):
        super(Block, self).__init__()
        self.padding = padding
        self.bottleneck = blockname
        self.stride = stride  
        blocks = []
        for i in range(num):
            mode = 'A' if i==0 else 'B'
            blocks.append(self._makeblock(mode, inplance, plance))
            inplance = plance*4
        self.blocks = nn.Sequential(*blocks)


    def _makeblock(self, mode, inplance, plance):
        reduce_channel = None
        if mode == 'A':
            reduce_channel = nn.Conv2d(inplance, plance*4, 1,stride = self.stride)
            return self.bottleneck(inplance,plance,stride=self.stride, reduce_channel = reduce_channel)
        else:
            return self.bottleneck(inplance,plance, stride=1, reduce_channel=reduce_channel)

    def forward(self, x):
        return self.blocks(x)


def Resnet_model(name='resnet101', **kwargs):
   model = Resnet(name, **kwargs)
   return model