import torch
import torch.nn as nn
from xception import XceptionNet
from vgg import vggnet
from resnet import Resnet_model
from aspp import Aspp
import torch.nn.functional as F

class DeeplabV3plus(nn.Module):
    def __init__(self, model_base, name=None, num_class = 10):
        super(DeeplabV3plus, self).__init__()
        if name is not None:
            self.model_base = model_base(name=name)
        else:
            self.model_base = model_base()
        self.encode1 = self.model_base.layers1
        self.encode2 = self.model_base.layers2
        self.encode3 = self.model_base.layers3
        self.encode4 = self.model_base.layers4
        self.encode5 = self.model_base.layers5

        self.aspp = Aspp(2048, 512)
        self.conv1x1 = nn.Conv2d(2048, 512, 1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv3x3 = nn.Conv2d(1024, num_class, 3, stride=1, padding=1)


    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]
        x = self.encode1(x)
        x = self.encode2(x)
        x = self.encode3(x)
        x = self.encode4(x)
        x = self.encode5(x)
        x1 = self.conv1x1(x)
        x2 = self.aspp(x)
        x = torch.cat([x1,x2], 1)
        x = self.conv3x3(x)
        x = F.upsample(x, size=(h, w), mode='bilinear')
        return x


def Xcept_dpv3plus(**kwargs):
    model = DeeplabV3plus(XceptionNet, **kwargs)
    return model

def vggunet(**kwargs):
    model = DeeplabV3plus(vggnet, **kwargs)
    return model

def resnetunet(**kwargs):
    model = DeeplabV3plus(Resnet_model, **kwargs)
    return model

if __name__ == '__main__':
    Xcept_model = Xcept_dpv3plus()
    x = torch.rand([1,3,224,224])
    out = Xcept_model(x)
    print(out.size())