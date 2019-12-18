import torch
import torch.nn as nn
from collections import OrderedDict


model_dict={
    'vgg16' :[64, 64, 'M', 128, 128, 'M', 256,256,256,'M',512,512,512,'M', 512,512,512,'M']
}


class VGG(nn.Module):
    def __init__(self, model, in_channel=3, num_class = 1000):
        super(VGG, self).__init__()

        layers = OrderedDict()

        i = 1
        j = 1
        in_channel = in_channel
        for layer in model_dict[model]:
            if layer != 'M':
                layers['Conv{}_{}'.format(i, j)] = nn.Conv2d(in_channel, layer, 3, stride=1, padding=1)
                layers['relu{}_{}'.format(i, j)] = nn.ReLU()
                in_channel = layer
                j += 1
            else:
                layers['pool{}'.format(i)] = nn.MaxPool2d(2, stride=2)
                exec("self.layers%s=nn.Sequential(layers)" % i)
                layers = OrderedDict()
                i += 1
                j = 1
        self.outchannel5 = 512
        self.outchannel4 = 512
        self.outchannel3 = 256
        self.outchannel2 = 128
        self.outchannel1 = 64
        self.GAP = nn.AdaptiveAvgPool2d(7)
        self.fc = nn.Linear(2048, num_class)

    def forward(self, x):
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)
        x = self.GAP(x)
        x.squeeze_()
        x = self.fc(x)
        return x


def vggnet(name = 'vgg16', **kwargs):
    model = VGG(name, **kwargs)
    return model
