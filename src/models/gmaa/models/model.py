from configparser import NoSectionError
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import functools


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self):
        self.apply(self._weights_init_fn)

    def _weights_init_fn(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:     
            m.weight.data.normal_(0.0, 0.02)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)         
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


class ResidualBlock(BaseNetwork):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

        self.init_weights()

    def forward(self, x):
        return x + self.main(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc) 
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)  
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class Generator(BaseNetwork):
    """Generator network."""

    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__() 

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7,
                                stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(
            conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # self.debug1 = nn.Sequential(*layers)

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2,
                                    kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(
                curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # self.debug2 = nn.Sequential(*layers)

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # self.debug3 = nn.Sequential(*layers)

        self.down = nn.Sequential(*layers)

        # Up-sampling layers.
        # Up-sampling layers of main1
        layers = []
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim //
                                             2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(
                curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2
        self.main1 = nn.Sequential(*layers)
        # Same architecture for the color regression 
        layers = []
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7,
                                stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.im_reg = nn.Sequential(*layers)
        # One Channel output and Sigmoid function for the attention layer  
        layers = []
        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7,
                                stride=1, padding=3, bias=False))
        layers.append(nn.Sigmoid())  # Values between 0 and 1
        self.im_att = nn.Sequential(*layers)

        # Up-sampling layers of main2
        layers = []
        curr_dim = curr_dim * 4
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim //
                                             2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(
                curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2
        self.main2 = nn.Sequential(*layers)
        # self.debug4 = nn.Sequential(*layers)
        # noise
        layers = []
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7,
                                stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.im_noise = nn.Sequential(*layers)


        self.init_weights()

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.unsqueeze(2).unsqueeze(3) # (16, 17)->(16, 17, 1, 1)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))  

        x = torch.cat([x, c], dim=1)   
        down = self.down(x) # down:torch.Size([2, 256, 32, 32])
        # reg and att
        up1 = self.main1(down) # torch.Size([2, 64, 128, 128])
        att = self.im_att(up1) # torch.Size([2, 1, 128, 128])
        reg = self.im_reg(up1) # torch.Size([2, 3, 128, 128])
        # noise
        up2 = self.main2(down)  # up2 torch.Size([2, 64, 128, 128])
        noise = self.im_noise(up2)
  
        return att, reg, noise  


class Generator_efgan(BaseNetwork):
    """Generator network."""

    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator_efgan, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7,
                                stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(
            conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        self.conv0 = nn.Sequential(*layers)
        # layers = []

        # Same architecture(conv0) for the color regression
        layers = []
        # Down-sampling layers. Down 1/4
        curr_dim = conv_dim # 64
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2,
                                    kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(
                curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2 # 128 256
        
        self.convs1 = nn.Sequential(*layers)
        # layers = []

        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        self.rbs2 = nn.Sequential(*layers)
        # layers = []

        # Up-sampling layers. 
        layers.append(nn.ConvTranspose2d(curr_dim, curr_dim,
                                        kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(
            curr_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
    
        self.transconv_up3 = nn.Sequential(*layers)
        # layers = []

        #TODO torch.nn.Upsample

        layers.append(nn.Conv2d(curr_dim, curr_dim // 2,
                                kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(
            curr_dim // 2, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        curr_dim = curr_dim // 2 # 128

        self.conv4 = nn.Sequential(*layers)
        # layers = []

        # Up-sampling layers. 
        layers.append(nn.ConvTranspose2d(curr_dim, curr_dim,
                                        kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(
            curr_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
    
        self.transconv_up5 = nn.Sequential(*layers)
        # layers = []

        layers.append(nn.Conv2d(curr_dim, curr_dim // 2,
                                kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(
            curr_dim // 2, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        curr_dim = curr_dim // 2 # 64

        self.conv6 = nn.Sequential(*layers)
        # layers = []

        layers.append(nn.Conv2d(curr_dim, 3,
                                kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.Tanh())
        self.im_reg = nn.Sequential(*layers)

        # from same conv0. Another Channel with Sigmoid function for the attention layer
        layers = []

        for i in range(repeat_num // 3): # default 6/3=2
            layers.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))
        
        layers.append(nn.Conv2d(curr_dim, 1,
                                kernel_size=3, stride=1, padding=1, bias=False))

        layers.append(nn.Sigmoid())  # Values between 0 and 1
        self.im_att = nn.Sequential(*layers)

        self.init_weights()

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.

        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))

        x = torch.cat([x, c], dim=1)
        features = self.conv0(x)

        reg = self.im_reg(features)
        att = self.im_att(features)

        return att, reg


class Generator_pre(BaseNetwork):
    """Generator network."""

    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator_pre, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7,
                                stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(
            conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        self.debug1 = nn.Sequential(*layers)

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2,
                                    kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(
                curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        self.debug2 = nn.Sequential(*layers)

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        self.debug3 = nn.Sequential(*layers)

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim //
                                             2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(
                curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.main = nn.Sequential(*layers)

        self.debug4 = nn.Sequential(*layers)

        # Same architecture for the color regression 
        layers = []
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7,
                                stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.im_reg = nn.Sequential(*layers)

        # One Channel output and Sigmoid function for the attention layer  
        layers = []
        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7,
                                stride=1, padding=3, bias=False))
        layers.append(nn.Sigmoid())  # Values between 0 and 1
        self.im_att = nn.Sequential(*layers)

        self.init_weights()

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.unsqueeze(2).unsqueeze(3) # (16, 17)->(16, 17, 1, 1)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3)) # (16, 17, 1, 1)->(16, 17, 128, 128)

        x = torch.cat([x, c], dim=1) # 16, 17+3, 128, 128)
        features = self.main(x) # features(16, 64, 128, 128)

        reg = self.im_reg(features) # reg(16, 3, 128, 128)
        att = self.im_att(features) # att(16, 1, 128, 128)

        return att, reg  


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


def make_layer(block, n_layers): 
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class H_RRDB(nn.Module):
    def __init__(self, in_nc=3, nf=64, nb=23, gc=32):
        super(H_RRDB, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc) 

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb) 
        self.trunk_conv = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)
        self.tanh = nn.Tanh() 

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        out = self.tanh(trunk) 
        return out


class Discriminator(BaseNetwork):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=9): # 6->8 image_size: 128->512
        super(Discriminator, self).__init__()

        layers = []
        layers.append(
            nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2,
                                    kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(
            curr_dim, c_dim, kernel_size=kernel_size, bias=False)

        self.init_weights()

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h) 
        out_cls = self.conv2(h)

        return out_src.squeeze(), out_cls.squeeze(-1).squeeze(-1)  

