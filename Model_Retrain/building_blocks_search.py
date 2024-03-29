# -*- coding: utf-8 -*-
# @Date    : 2019-08-02
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

from torch import nn
import torch.nn.functional as F
from TransformerBlock import TransEncoderBlock

# search space
CONV_TYPE = {0: 'conv', 1: 'trans'}   #'transconv'
NORM_TYPE = {0: None, 1: 'bn', 2: 'in'}
UP_TYPE = {0: 'bilinear', 1: 'nearest', 2: 'deconv'}
SHORT_CUT_TYPE = {0: False, 1: True}
SKIP_TYPE = {0: False, 1: True}


def decimal2binary(n):
    return bin(n).replace("0b", "")

class Con_vBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_block, ksize=3):
        super(Con_vBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.inn = nn.InstanceNorm2d(in_channels)
        self.up_block = up_block
        self.deconv = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels, ksize, padding=ksize//2)

    def set_arch(self, up_id, norm_id):
        self.up_type = UP_TYPE[up_id]
        self.norm_type = NORM_TYPE[norm_id]

    def forward(self, x):
        # norm
        if self.norm_type:
            if self.norm_type == 'bn':
                h = self.bn(x)
            elif self.norm_type == 'in':
                h = self.inn(x)
            else:
                raise NotImplementedError(self.norm_type)
        else:
            h = x

        # activation
        h = nn.ReLU()(h)

        # whether this is a upsample block
        if self.up_block:
            if self.up_type == 'deconv':
                h = self.deconv(h)
            else:
                h = F.interpolate(h, scale_factor=2, mode=self.up_type)

        # conv
        out = self.conv(h)
        return out


class TransBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_block, hw, ksize=3):
        super(TransBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.bn = nn.BatchNorm2d(in_channels)
        self.inn = nn.InstanceNorm2d(in_channels)
        self.up_block = up_block
        self.deconv = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2)

        self.transformer_encoder = TransEncoderBlock(dim=(2 * hw) ** 2)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=(2*hw) ** 2, nhead=8)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def set_arch(self, up_id, norm_id):
        self.up_type = UP_TYPE[up_id]
        self.norm_type = NORM_TYPE[norm_id]

    def forward(self, x):
        # norm
        if self.norm_type:
            if self.norm_type == 'bn':
                h = self.bn(x)
            elif self.norm_type == 'in':
                h = self.inn(x)
            else:
                raise NotImplementedError(self.norm_type)
        else:
            h = x

        # activation
        h = nn.ReLU()(h)

        # whether this is a upsample block
        if self.up_block:
            if self.up_type == 'deconv':
                h = self.deconv(h)
            else:
                h = F.interpolate(h, scale_factor=2, mode=self.up_type)

        h_ = h.view(-1, self.in_channels, h.size()[-1]**2)
        # h_ = h_.transpose(1,2).contiguous()
        trans_h = self.transformer_encoder(h_)
        trans_h = trans_h.view(-1, self.in_channels, h.size()[-1], h.size()[-1])
        # trans_h = trans_h.transpose(1,2)

        out = self.conv1x1(trans_h).contiguous()
        return out


class TransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_block, hw, ksize=3):
        super(TransConvBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.bn = nn.BatchNorm2d(in_channels)
        self.inn = nn.InstanceNorm2d(in_channels)
        self.up_block = up_block
        self.deconv = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2)

        self.transformer_encoder = TransEncoderBlock(dim=(2*hw) ** 2)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=ksize//2)

    def set_arch(self, up_id, norm_id):
        self.up_type = UP_TYPE[up_id]
        self.norm_type = NORM_TYPE[norm_id]

    def forward(self, x):
        # norm
        if self.norm_type:
            if self.norm_type == 'bn':
                h = self.bn(x)
            elif self.norm_type == 'in':
                h = self.inn(x)
            else:
                raise NotImplementedError(self.norm_type)
        else:
            h = x

        # activation
        h = nn.ReLU()(h)

        # whether this is a upsample block
        if self.up_block:
            if self.up_type == 'deconv':
                h = self.deconv(h)
            else:
                h = F.interpolate(h, scale_factor=2, mode=self.up_type)

        h_ = h.view(-1, self.in_channels, h.size()[-1] ** 2)
        trans_h = self.transformer_encoder(h_)
        trans_h = trans_h.view(-1, self.in_channels, h.size()[-1], h.size()[-1])
        out = self.conv(trans_h)
        return out


class Cell(nn.Module):
    def __init__(self, in_channels, out_channels, num_skip_in, hw, ksize=3):
        super(Cell, self).__init__()

        self.conv1 = Con_vBlock(in_channels, out_channels, ksize=ksize, up_block=True)
        self.trans1 = TransBlock(in_channels, out_channels, ksize=ksize, hw=hw, up_block=True)
        # self.transconv1 = TransConvBlock(in_channels, out_channels, ksize=ksize, hw=hw, up_block=True)

        self.conv2 = Con_vBlock(out_channels, out_channels, ksize=ksize, up_block=False)
        self.trans2 = TransBlock(out_channels, out_channels, ksize=ksize, hw=hw, up_block=False)
        # self.transconv2 = TransConvBlock(out_channels, out_channels, ksize=ksize, hw=hw, up_block=False)

        self.deconv_sc = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2)
        self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # skip_in without channel up
        self.skip_deconvx2 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.skip_deconvx4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels*2, in_channels*2, kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels*2, in_channels*2, kernel_size=2, stride=2)
        )
        self.skip_deconvx8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels*4, in_channels*4, kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels*4, in_channels*4, kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels*4, in_channels*4, kernel_size=2, stride=2)
        )
        # skip_in with channel up
        self.num_skip_in = num_skip_in
        if num_skip_in:
            new_in_channels = in_channels * 2**(num_skip_in-1)  # reset to 256
            self.skip_in_ops = nn.ModuleList([nn.Conv2d(new_in_channels//(2**i), out_channels, kernel_size=1) for i in range(num_skip_in)])

    def set_arch(self, conv_id, norm_id, up_id, short_cut_id, skip_ins):
        self.conv_type = CONV_TYPE[conv_id]
        self.up_type = UP_TYPE[up_id]
        self.short_cut = SHORT_CUT_TYPE[short_cut_id]

        self.conv1.set_arch(up_id, norm_id)
        self.trans1.set_arch(up_id, norm_id)
        # self.transconv1.set_arch(up_id, norm_id)
        self.conv2.set_arch(up_id, norm_id)
        self.trans2.set_arch(up_id, norm_id)
        # self.transconv2.set_arch(up_id, norm_id)

        if self.num_skip_in:
            self.skip_ins = [0 for _ in range(self.num_skip_in)]
            for skip_idx, skip_in in enumerate(decimal2binary(skip_ins)[::-1]):
                self.skip_ins[-(skip_idx + 1)] = int(skip_in)


    def forward(self, x, skip_ft=None):
        residual = x

        # first conv
        if self.conv_type == 'conv':
            h = self.conv1(residual)
        elif self.conv_type == 'trans':
            h = self.trans1(residual)
        # elif self.conv_type == 'transconv':
        #     h = self.transconv1(residual)
        else:
            raise NotImplementedError(self.norm_type)
        _, _, ht, wt = h.size()
        h_skip_out = h
        # second conv
        if self.num_skip_in:
            assert len(self.skip_in_ops) == len(self.skip_ins)
            for skip_flag, ft, skip_in_op in zip(self.skip_ins, skip_ft, self.skip_in_ops):
                if skip_flag:
                    if self.up_type != 'deconv':
                        skip_h = F.interpolate(ft, size=(ht, wt), mode=self.up_type)
                        h += skip_in_op(skip_h)
                    else:
                        scale = wt // ft.size()[-1]
                        up_type = 'skip_deconvx' + str(scale)
                        h += skip_in_op(getattr(self, up_type)(ft))

        if self.conv_type == 'conv':
            final_out = self.conv2(h)
        elif self.conv_type == 'trans':
            final_out = self.trans2(h)
        # elif self.conv_type == 'transconv':
        #     final_out = self.transconv2(h)
        else:
            raise NotImplementedError(self.norm_type)

        # shortcut
        if self.short_cut:
            if self.up_type != 'deconv':
                final_out += self.c_sc(F.interpolate(x, scale_factor=2, mode=self.up_type))
            else:
                final_out += self.c_sc(self.deconv_sc(x))

        return h_skip_out, final_out


