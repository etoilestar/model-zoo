import torch
import torch.nn as nn

def Conv1x1(in_channel, out_channel, stride=1, padding = 0,dilation=1,groups=1):
    return nn.Conv2d(in_channel, out_channel, 1, stride=stride, padding=padding,dilation=dilation,groups=groups)

def Conv3x3(in_channel, out_channel, stride=1, padding = 0, dilation=1, groups = 1):
    return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=padding, dilation=dilation,groups=groups)

def make_layers(inchannel, outchannel, last_stride = 2):
    return nn.Sequential(
        Conv3x3(inchannel, inchannel, stride=1, padding=1, groups=inchannel),
        Conv1x1(inchannel, outchannel, stride=1),
        nn.BatchNorm2d(outchannel),
        nn.ReLU(inplace=True),
        Conv3x3(outchannel, outchannel, stride=1, padding=1, groups=outchannel),
        nn.BatchNorm2d(outchannel),
        nn.ReLU(inplace=True),
        Conv3x3(outchannel, outchannel, stride=last_stride, padding=1, groups=outchannel),
        nn.BatchNorm2d(outchannel),
        nn.ReLU(inplace=True)
    )

class Entry_FLow_Blocks(nn.Module):
    def __init__(self, channels):
        super(Entry_FLow_Blocks, self).__init__()
        self.EntryBlock1 = make_layers(channels[0], channels[1])
        self.EntryBlock2 = make_layers(channels[1], channels[2])
        self.EntryBlock3 = make_layers(channels[2], channels[3])

        self.Conv1 = Conv1x1(channels[0],channels[1],stride=2)
        self.Conv2 = Conv1x1(channels[1],channels[2],stride=2)
        self.Conv3 = Conv1x1(channels[2],channels[3],stride=2)

    def forward(self, x):
        x_c = self.Conv1(x)
        x = self.EntryBlock1(x)
        x = torch.add(x, x_c)

        x_c = self.Conv2(x)
        x = self.EntryBlock2(x)
        x = torch.add(x, x_c)

        x_c = self.Conv3(x)
        x = self.EntryBlock3(x)
        x = torch.add(x, x_c)
        return x


class Middle_FLow_Blocks(nn.Module):
    def __init__(self, channels, num):
        super(Middle_FLow_Blocks, self).__init__()
        Blocks = []
        for i in range(num):
            Blocks.append(make_layers(channels[0], channels[0], last_stride=1))
        self.MiddleBLocks = nn.Sequential(*Blocks)

    def forward(self, x):
        return self.MiddleBLocks(x)


class Exit_FLow_Blocks(nn.Module):
    def __init__(self, channels):
        super(Exit_FLow_Blocks, self).__init__()
        self.Block = make_layers(channels[0], channels[1])
        self.conv = Conv1x1(728, 1024,stride=2)

    def forward(self, x):
        input = self.conv(x)
        return input+self.Block(x)

class Xception(nn.Module):
    def __init__(self, in_channel=3, num_class = 1000):
        super(Xception, self).__init__()
        self.layers1 = nn.Sequential(
            Conv3x3(in_channel, 32, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Conv3x3(32, 64, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layers2 = Entry_FLow_Blocks([64, 128, 256, 728])
        self.layers3 = Middle_FLow_Blocks([728], 5)
        self.layers4 = Exit_FLow_Blocks([728,1024])
        self.layers5 = nn.Sequential(
            Conv3x3(1024, 1024, stride=1, padding=1,groups = 1024),
            Conv1x1(1024, 2048),
            # nn.BatchNorm2d(1024),
            # nn.ReLU(inplace=True),
            # Conv3x3(1024, 2048, stride=1, padding=2048),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )

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

def XceptionNet(**kwargs):
    model = Xception(**kwargs)
    return model
