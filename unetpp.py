import torch.nn as nn
import torch
from vgg import vggnet
from resnet import Resnet_model

class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel, up_smple_mode='Bilinear', with_conv_channels=None):
        super(UpSample, self).__init__()
        self.with_conv_channels = with_conv_channels
        if up_smple_mode=='Bilinear':
            self.upsample = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_channel, out_channel, 1)
            )
        elif up_smple_mode=='TransposeConv':
            self.upsample = nn.ConvTranspose1d(in_channel, out_channel, 2, stride=2)
        if self.with_conv_channels is not None:
            self.conv = nn.Conv2d(with_conv_channels[0], with_conv_channels[1], 1)

    def forward(self, x, map_ad=None):
        x = self.upsample(x)
        if self.with_conv_channels is not None:
            assert map_ad is not None
            for map in map_ad:
                x = torch.cat((x, map), axis=1)
            x = self.conv(x)
        return x

class Unetpp(nn.Module):
    def __init__(self, model_base, name='vgg16', class_num=10):
        super(Unetpp, self).__init__()
        self.model_base = model_base(name=name)
        self.encode1 = self.model_base.layers1
        self.encode2 = self.model_base.layers2
        self.encode3 = self.model_base.layers3
        self.encode4 = self.model_base.layers4
        self.encode5 = self.model_base.layers5

        self.center1 = nn.Conv2d(self.model_base.outchannel5, 1024, 1, padding=0,stride=1)
        self.center2 = nn.Conv2d(1024, 1024, 1, padding=0, stride=1)

        self.decode5 = UpSample(1024, 512, with_conv_channels=[512+self.model_base.outchannel4, 512])

        self.decode4 = UpSample(512, 256, with_conv_channels=[256+256+self.model_base.outchannel3, 256])
        self.decode4_1 = UpSample(self.model_base.outchannel4, 256, with_conv_channels=[256+self.model_base.outchannel3, 256])

        self.decode3 = UpSample(256, 128, with_conv_channels=[128+128+128+self.model_base.outchannel2, 128])
        self.decode3_1 = UpSample(self.model_base.outchannel3, 128, with_conv_channels=[128+self.model_base.outchannel2, 128])
        self.decode3_2 = UpSample(256, 128, with_conv_channels=[128+128+self.model_base.outchannel2, 128])

        self.decode2 = UpSample(128, 64, with_conv_channels=[64+64+64+64+self.model_base.outchannel1, 64])
        self.decode2_1 = UpSample(self.model_base.outchannel2, 64, with_conv_channels=[64+self.model_base.outchannel1, 64])
        self.decode2_2 = UpSample(128, 64, with_conv_channels=[64+64+self.model_base.outchannel1, 64])
        self.decode2_3 = UpSample(128, 64, with_conv_channels=[64+64+64+self.model_base.outchannel1, 64])

        self.decode1 = UpSample(64, 64)

        self.final = nn.Sequential(
            nn.Conv2d(64, class_num, 1),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

    def forward(self, x):
        x1 = self.encode1(x)
        x2 = self.encode2(x1)
        x3 = self.encode3(x2)
        x4 = self.encode4(x3)
        x5 = self.encode5(x4)

        x5 = self.center1(x5)
        x5 = self.center2(x5)

        x4_ = self.decode5(x5, (x4,))

        x3_1 = self.decode4_1(x4, (x3,))
        x3_ = self.decode4(x4_, (x3, x3_1))

        x2_1 = self.decode3_1(x3, (x2,))
        x2_2 = self.decode3_2(x3_1, (x2, x2_1))
        x2_ = self.decode3(x3_, (x2, x2_1, x2_2))

        x1_1 = self.decode2_1(x2, (x1,))
        x1_2 = self.decode2_2(x2_1, (x1, x1_1))
        x1_3 = self.decode2_3(x2_2, (x1, x1_1, x1_2))
        x1_ = self.decode2(x2_, (x1, x1_1, x1_2, x1_3))

        xout = self.decode1(x1_)
        out = self.final(xout)
        return out

def vggunetpp(**kwargs):
    model = Unetpp(vggnet, **kwargs)
    return model

def resnetunetpp(**kwargs):
    model = Unetpp(Resnet_model, **kwargs)
    return model

if __name__ == '__main__':
#    Res_Unet = resnetunetpp(name='resnet101')
    VGG_Unet = vggunetpp(name='vgg16')
    x = torch.rand([1,3,224,224])
#    out = Res_Unet(x)
    out = VGG_Unet(x)
    print(out.size())