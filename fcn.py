import torch
import torch.nn as nn
from vgg import vggnet
from resnet import Resnet_model

class FCN(nn.Module):
    def __init__(self, model_base, name='vgg16', fcn_class='8s', class_num = 10):
        super(FCN, self).__init__()
        self. model_base = model_base(name=name)
        self.name = name
        self.fcn_name = fcn_class

#        self.model_base.layers1.Conv1_1.padding = (100,100)

        self.encode1 = self.model_base.layers1
        self.encode2 = self.model_base.layers2
        self.encode3 = self.model_base.layers3
        self.encode4 = self.model_base.layers4
        self.encode5 = self.model_base.layers5

        fcn_inchannel4 = self. model_base.outchannel4

        self.conv4 = nn.Conv2d(fcn_inchannel4, 4096, 1)
        self.conv5 = nn.Conv2d(4096, 4096, 1)
        self.conv6 = nn.Conv2d(4096, class_num, 1)
        self.conv1x1 = nn.Conv2d(class_num, class_num, 1)
        self.decode32s = nn.ConvTranspose2d(class_num, class_num, 32,stride=32,bias=False)

        fcn_inchannel3 = self.model_base.outchannel3
        self.conv3 = nn.Conv2d(fcn_inchannel3, class_num, 1)
        self.decode16s_1 = nn.ConvTranspose2d(class_num, class_num, 2, stride=2)
        self.decode16s_2 = nn.ConvTranspose2d(class_num, class_num, 16, stride=16)

        fcn_inchannel2 = self.model_base.outchannel2
        self.conv2 = nn.Conv2d(fcn_inchannel2, class_num, 1)
        self.decode8s_1 = nn.ConvTranspose2d(class_num, class_num, 2,stride=2)
        self.decode8s_2 = nn.ConvTranspose2d(class_num, class_num, 8,stride=8)

    def forward(self, x):
        x = self.encode1(x)
        x2 = self.encode2(x)
        x3 = self.encode3(x2)
        x4 = self.encode4(x3)
        x5 = self.encode5(x4)
        x5 = self.conv4(x5)
        x5 = self.conv5(x5)
        x5 = self.conv6(x5)
        x4 = self.conv3(x4)
        x3 = self.conv2(x3)
        if self.fcn_name == '32s':
            return self.decode32s(x5)
        else:
            x5 = self.decode16s_1(x5)
            if self.fcn_name == '16s':
                return self.decode16s_2(self.conv1x1(x4+x5))
            elif self.fcn_name == '8s':
                x3plus4 = self.decode8s_1(self.conv1x1(x4+x5))
                return self.decode8s_2(self.conv1x1(x3+x3plus4))


def vggfcn(**kwargs):
    model = FCN(vggnet, **kwargs)
    return model

def resnetfcn(**kwargs):
    model = FCN(Resnet_model, **kwargs)
    return model

if __name__ == '__main__':
    VGG_FCN = resnetfcn(name='resnet101')
    # VGG_FCN = vggfcn()
    x = torch.rand([1,3,224,224])
    out = VGG_FCN(x)
    print(out.size())