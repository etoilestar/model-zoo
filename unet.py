import torch.nn as nn
import torch
from vgg import vggnet
from resnet import Resnet_model

class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel, up_smple_mode='Bilinear'):
        super(UpSample, self).__init__()
        if up_smple_mode=='Bilinear':
            self.upsample = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_channel, out_channel, 1)
            )
        elif up_smple_mode=='TransposeConv':
            self.upsample = nn.ConvTranspose1d(in_channel, out_channel, 2, stride=2)

    def forward(self, x):
        return self.upsample(x)

class Unet(nn.Module):
    def __init__(self, model_base, name='vgg16', class_num=10):
        super(Unet, self).__init__()
        self.model_base = model_base(name=name)
        self.encode1 = self.model_base.layers1
        self.encode2 = self.model_base.layers2
        self.encode3 = self.model_base.layers3
        self.encode4 = self.model_base.layers4
        self.encode5 = self.model_base.layers5

        self.center1 = nn.Conv2d(self.model_base.outchannel5, 1024, 1, padding=0,stride=1)
        self.center2 = nn.Conv2d(1024, 1024, 1, padding=0, stride=1)
        self.decode1 = UpSample(1024, 512)
        self.decode2 = UpSample(512+self.model_base.outchannel4, 256)
        self.decode3 = UpSample(256+self.model_base.outchannel3, 64)
        self.decode4 = UpSample(64+self.model_base.outchannel2, 64)
        self.decode5 = UpSample(64, 64)
        self.final = nn.Sequential(
            nn.Conv2d(64, class_num, 1),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

    def forward(self, x):
        x = self.encode1(x)
        x1 = self.encode2(x)
        x2 = self.encode3(x1)
        x3 = self.encode4(x2)
        x4 = self.encode5(x3)

        x4 = self.center1(x4)
        x4 = self.center2(x4)

        x3_ = self.decode1(x4)
        x2_ = self.decode2(torch.cat((x3_, x3), 1))
        x1_ = self.decode3(torch.cat((x2_,x2), 1))
        x1_ = self.decode4(torch.cat((x1_,x1), 1))
        xout = self.decode5(x1_)
        out = self.final(xout)
        return out


def vggunet(**kwargs):
    model = Unet(vggnet, **kwargs)
    return model

def resnetunet(**kwargs):
    model = Unet(Resnet_model, **kwargs)
    return model

if __name__ == '__main__':
    Res_Unet = resnetunet(name='resnet101')
    VGG_Unet = vggunet(name='vgg16')
    x = torch.rand([1,3,224,224])
#    out = Res_Unet(x)
    out = VGG_Unet(x)
    print(out.size())
