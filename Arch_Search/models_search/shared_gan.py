# -*- coding: utf-8 -*-
# @Date    : 2019-08-15
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0
import torch.nn as nn
import torch
from torch.autograd import Variable
from models_search.building_blocks_search import Cell
import torch.nn.functional as F

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])

class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = 256
        self.c_dim = 100
        self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # if cfg.CUDA:
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        # else:
        #     eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        self.ch = args.gf_dim
        self.bottom_width = args.bottom_width

        self.ca_net = CA_NET()
        self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * args.gf_dim)
        # stage 1
        self.cell1 = Cell(args.gf_dim, args.gf_dim//2, num_skip_in=0, hw=4)
        self.cell2 = Cell(args.gf_dim//2, args.gf_dim//4, num_skip_in=1, hw=8)
        self.cell3 = Cell(args.gf_dim//4, args.gf_dim//8, num_skip_in=2, hw=16)
        self.cell4 = Cell(args.gf_dim//8, args.gf_dim//16, num_skip_in=3, hw=32)

        self.to_rgb8 = nn.Sequential(
            nn.BatchNorm2d(args.gf_dim//2),
            nn.ReLU(),
            nn.Conv2d(args.gf_dim//2, 3, 3, 1, 1),
            nn.Tanh()
        )
        self.to_rgb16 = nn.Sequential(
            nn.BatchNorm2d(args.gf_dim//4),
            nn.ReLU(),
            nn.Conv2d(args.gf_dim//4, 3, 3, 1, 1),
            nn.Tanh()
        )
        self.to_rgb32 = nn.Sequential(
            nn.BatchNorm2d(args.gf_dim//8),
            nn.ReLU(),
            nn.Conv2d(args.gf_dim//8, 3, 3, 1, 1),
            nn.Tanh()
        )
        self.to_rgb64 = nn.Sequential(
            nn.BatchNorm2d(args.gf_dim//16),
            nn.ReLU(),
            nn.Conv2d(args.gf_dim//16, 3, 3, 1, 1),
            nn.Tanh()
        )

    def set_arch(self, arch_id, cur_stage):
        if not isinstance(arch_id, list):
            arch_id = arch_id.to('cpu').numpy().tolist()
        arch_id = [int(x) for x in arch_id]
        self.cur_stage = cur_stage
        arch_stage1 = arch_id[:4]
        self.cell1.set_arch(conv_id=arch_stage1[0], norm_id=arch_stage1[1], up_id=arch_stage1[2],
                            short_cut_id=arch_stage1[3], skip_ins=[])
        if cur_stage >= 1:
            arch_stage2 = arch_id[4:9]
            self.cell2.set_arch(conv_id=arch_stage2[0], norm_id=arch_stage2[1], up_id=arch_stage2[2],
                                short_cut_id=arch_stage2[3], skip_ins=arch_stage2[4])

        if cur_stage >= 2:
            arch_stage3 = arch_id[9:14]
            self.cell3.set_arch(conv_id=arch_stage3[0], norm_id=arch_stage3[1], up_id=arch_stage3[2],
                                short_cut_id=arch_stage3[3], skip_ins=arch_stage3[4])
        
        if cur_stage == 3:
            arch_stage4 = arch_id[14:]
            self.cell4.set_arch(conv_id=arch_stage4[0], norm_id=arch_stage4[1], up_id=arch_stage4[2],
                            short_cut_id=arch_stage4[3], skip_ins=arch_stage4[4])
            
    def forward(self, sent_emb, noise):
        c_code, mu, logvar = self.ca_net(sent_emb)
        # print(sent_emb.shape)
        # print(c_code.shape)
        z = torch.cat((c_code, noise), 1)
        h = self.l1(z).view(-1, self.ch, self.bottom_width, self.bottom_width)

        h1_skip_out, h1 = self.cell1(h)
        if self.cur_stage == 0:
            return self.to_rgb8(h1)
        # print("h1: ", h1.shape,h1_skip_out.shape)
        h2_skip_out, h2 = self.cell2(h1, (h1_skip_out,))
        if self.cur_stage == 1:
            return self.to_rgb16(h2)
        h3_skip_out, h3 = self.cell3(h2, (h1_skip_out, h2_skip_out))
        if self.cur_stage == 2:
            return self.to_rgb32(h3)
        _, h4 = self.cell4(h3, (h1_skip_out, h2_skip_out, h3_skip_out))
        if self.cur_stage == 3:
            return self.to_rgb64(h4)


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)


class OptimizedDisBlock(nn.Module):
    def __init__(
            self,
            args,
            in_channels,
            out_channels,
            ksize=3,
            pad=1,
            activation=nn.ReLU()):
        super(OptimizedDisBlock, self).__init__()
        self.activation = activation

        self.c1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            padding=pad)
        self.c2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=ksize,
            padding=pad)
        self.c_sc = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0)
        if args.d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)
            self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class DisBlock(nn.Module):
    def __init__(
            self,
            args,
            in_channels,
            out_channels,
            hidden_channels=None,
            ksize=3,
            pad=1,
            activation=nn.ReLU(),
            downsample=False):
        super(DisBlock, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels

        self.c1 = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=ksize,
            padding=pad)
        self.c2 = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=ksize,
            padding=pad)
        if args.d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0)
            if args.d_spectral_norm:
                self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class Discriminator(nn.Module):
    def __init__(self, args, activation=nn.ReLU()):
        super(Discriminator, self).__init__()
        self.ch = args.df_dim
        self.activation = activation
        self.block1 = OptimizedDisBlock(args, 3, self.ch)
        self.block2 = DisBlock(
            args,
            self.ch,
            self.ch,
            activation=activation,
            downsample=True)
        self.block3 = DisBlock(
            args,
            self.ch,
            self.ch,
            activation=activation,
            downsample=False)
        self.block4 = DisBlock(
            args,
            self.ch,
            self.ch,
            activation=activation,
            downsample=False)
        self.l5 = nn.Linear(self.ch, 1, bias=False)
        if args.d_spectral_norm:
            self.l5 = nn.utils.spectral_norm(self.l5)
        self.cur_stage = 0

    def forward(self, x):
        h = x
        layers = [self.block1, self.block2, self.block3]
        variable_model = nn.Sequential(*layers[:(self.cur_stage + 1)])
        h = variable_model(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l5(h)

        return output
