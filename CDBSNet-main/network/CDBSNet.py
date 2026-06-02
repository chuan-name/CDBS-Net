import torch
import torch.nn as nn
import torch.nn.functional as F
from network.CDBSNet.swintransformer import swin_small_patch4_window7_224
from network.CDBSNet.block import  HybridCrossScanBlock, CrossDirectionalBoundaryGating, MSDG


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SEWeightModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)
        return weight * x


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)
        return output


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        self.bn_acti = bn_acti
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.bn_acti:
            self.bn_relu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)
        if self.bn_acti:
            output = self.bn_relu(output)
        return output


class aggregation(nn.Module):
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample7 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample8 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_upsample9 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4 * channel, 4 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(4 * channel, 4 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(4 * channel, 1, 1)

    def forward(self, x1, x2, x3, x4):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3
        x4_1 = self.conv_upsample6(self.upsample(self.upsample(self.upsample(x1)))) * \
               self.conv_upsample7(self.upsample(self.upsample(x2))) * \
               self.conv_upsample8(self.upsample(x3)) * x4

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x4_2 = torch.cat((x4_1, self.conv_upsample9(self.upsample(x3_2))), 1)
        x4_2 = self.conv_concat4(x4_2)

        x = self.conv4(x4_2)
        x = self.conv5(x)

        return x


class RrConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = Conv(in_channel, out_channel, 3, 1, 1, dilation=(1, 1), groups=1, bn_acti=True, bias=False)
        self.res_conv = nn.Conv2d(in_channel, out_channel, 1)
        self.res_back_conv = nn.Conv2d(out_channel, in_channel, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.conv(x)
        f1 = self.relu(x + self.res_conv(identity))
        f2 = torch.mul(torch.sigmoid(self.res_back_conv(f1)) + 1, identity)
        x = self.conv(f2)
        x = self.relu(self.res_conv(f2) + x)
        return x


class CDBSNet(nn.Module):

    def __init__(self, channel=32, pretrained=True):
        super(CDBSNet, self).__init__()


        self.swintransformer = swin_small_patch4_window7_224(1000, pretrained=pretrained)

        self.conv1 = Conv(96, 96, 3, 1, 1, dilation=(1, 1), groups=1, bn_acti=True, bias=False)
        self.conv2 = Conv(192, 192, 3, 1, 1, dilation=(1, 1), groups=1, bn_acti=True, bias=False)
        self.conv3 = Conv(384, 384, 3, 1, 1, dilation=(1, 1), groups=1, bn_acti=True, bias=False)
        self.conv4 = Conv(768, 768, 3, 1, 1, dilation=(1, 1), groups=1, bn_acti=True, bias=False)


        self.enhance1 = HybridCrossScanBlock(dim=96)
        self.enhance2 = HybridCrossScanBlock(dim=192)
        self.enhance3 = HybridCrossScanBlock(dim=384)

        self.cdbg1 = CrossDirectionalBoundaryGating(in_chs=96, out_chs=96)
        self.cdbg2 = CrossDirectionalBoundaryGating(in_chs=192, out_chs=192)
        self.cdbg3 = CrossDirectionalBoundaryGating(in_chs=384, out_chs=384)

        self.msdg= MSDG(dim=768)


        self.reduce1 = nn.Conv2d(96, channel, kernel_size=1)
        self.reduce2 = nn.Conv2d(192, channel, kernel_size=1)
        self.reduce3 = nn.Conv2d(384, channel, kernel_size=1)
        self.reduce4 = nn.Conv2d(768, channel, kernel_size=1)


        self.agg1 = aggregation(channel)


        self.ra1_conv1 = RrConv(768, 32)
        self.ra1_conv2 = RrConv(32, 32)
        self.ra1_conv3 = RrConv(32, 32)
        self.ra1_conv4 = RrConv(32, 32)
        self.ra1_conv5 = Conv(32, 1, 3, 1, 1, dilation=(1, 1), groups=1, bn_acti=True, bias=False)

        self.ra2_conv1 = RrConv(384, 32)
        self.ra2_conv2 = RrConv(32, 32)
        self.ra2_conv3 = RrConv(32, 32)
        self.ra2_conv4 = RrConv(32, 32)
        self.ra2_conv5 = Conv(32, 1, 3, 1, 1, dilation=(1, 1), groups=1, bn_acti=True, bias=False)

        self.ra3_conv1 = RrConv(192, 32)
        self.ra3_conv2 = RrConv(32, 32)
        self.ra3_conv3 = RrConv(32, 32)
        self.ra3_conv4 = RrConv(32, 32)
        self.ra3_conv5 = Conv(32, 1, 3, 1, 1, dilation=(1, 1), groups=1, bn_acti=True, bias=False)

        self.ra4_conv1 = RrConv(96, 32)
        self.ra4_conv2 = RrConv(32, 32)
        self.ra4_conv3 = RrConv(32, 32)
        self.ra4_conv4 = RrConv(32, 32)
        self.ra4_conv5 = Conv(32, 1, 3, 1, 1, dilation=(1, 1), groups=1, bn_acti=True, bias=False)

    def forward(self, x):
        B = x.shape[0]


        x, H, W = self.swintransformer.patch_embed(x)
        H1, W1 = H, W
        x = self.swintransformer.pos_drop(x)
        x_list = [x]
        x1_list = []

        for i, layer in enumerate(self.swintransformer.layers):
            x1, x, H, W = layer(x_list[i], H, W)
            x1_list.append(x1)
            x_list.append(x)

        x1 = x1_list[0].permute(0, 2, 1).view(B, 96, H1, W1)
        x2 = x1_list[1].permute(0, 2, 1).view(B, 192, H1 // 2, W1 // 2)
        x3 = x1_list[2].permute(0, 2, 1).view(B, 384, H1 // 4, W1 // 4)
        x4 = x1_list[3].permute(0, 2, 1).view(B, 768, H1 // 8, W1 // 8)


        swin_out1 = x1
        swin_out2 = x2
        swin_out3 = x3


        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)

        x1 = self.enhance1(x1)
        x2 = self.enhance2(x2)
        x3 = self.enhance3(x3)
        skip_list = [x1, x2, x3, x4]
        x4_fused = self.msdg(x4, skip_list)
        x4 = x4 + x4_fused

        x1_f = self.reduce1(x1)
        x2_f = self.reduce2(x2)
        x3_f = self.reduce3(x3)
        x4_f = self.reduce4(x4)


        x_agg = self.agg1(x4_f, x3_f, x2_f, x1_f)
        pre_res = F.interpolate(x_agg, scale_factor=4, mode='bilinear')


        d4 = self.ra1_conv1(x4)
        d4 = self.ra1_conv2(d4)
        d4 = self.ra1_conv3(d4)
        d4 = self.ra1_conv4(d4)
        lateral_map_4 = F.interpolate(self.ra1_conv5(d4), scale_factor=32, mode='bilinear')

        d3 = self.ra2_conv1(x3)
        d3 = self.ra2_conv2(d3)
        d3 = self.ra2_conv3(d3)
        d3 = self.ra2_conv4(d3)
        lateral_map_3 = F.interpolate(self.ra2_conv5(d3), scale_factor=16, mode='bilinear')

        d2 = self.ra3_conv1(x2)
        d2 = self.ra3_conv2(d2)
        d2 = self.ra3_conv3(d2)
        d2 = self.ra3_conv4(d2)
        lateral_map_2 = F.interpolate(self.ra3_conv5(d2), scale_factor=8, mode='bilinear')

        d1 = self.ra4_conv1(x1)
        d1 = self.ra4_conv2(d1)
        d1 = self.ra4_conv3(d1)
        d1 = self.ra4_conv4(d1)
        lateral_map_1 = F.interpolate(self.ra4_conv5(d1), scale_factor=4, mode='bilinear')


        return (
            pre_res,
            lateral_map_4,
            lateral_map_3,
            lateral_map_2,
            lateral_map_1,
        )
