import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsample1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1x1(x)
        return x


class Downsample2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1)
        self.conv1x1 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv1x1(x)
        return x


class Downsample3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=3, stride=2, padding=1)
        self.conv1x1 = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv1x1(x)
        return x


class CrossDirectionalBoundaryGating(nn.Module):

    def __init__(self, in_chs, out_chs):
        super(CrossDirectionalBoundaryGating, self).__init__()

        self.horizontal_conv = nn.Sequential(
            nn.Conv2d(in_chs, in_chs // 2, kernel_size=(1, 21), padding=(0, 10), bias=False),
            nn.BatchNorm2d(in_chs // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_chs // 2, out_chs // 2, kernel_size=3, padding=1, bias=False)
        )


        self.vertical_conv = nn.Sequential(
            nn.Conv2d(in_chs, in_chs // 2, kernel_size=(21, 1), padding=(10, 0), bias=False),
            nn.BatchNorm2d(in_chs // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_chs // 2, out_chs // 2, kernel_size=3, padding=1, bias=False)
        )

        self.boundary_extractor = nn.Sequential(
            nn.Conv2d(in_chs, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)


        self.fuse_h = nn.Conv2d(out_chs // 2 + 1, out_chs // 2, 1)
        self.fuse_v = nn.Conv2d(out_chs // 2 + 1, out_chs // 2, 1)
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_chs, out_chs, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        device = x.device


        x_h = self.horizontal_conv(x)
        x_v = self.vertical_conv(x)

        b_map = self.boundary_extractor(x)

        edge_v = F.conv2d(b_map, self.sobel_x.to(device), padding=1)
        edge_v = torch.abs(edge_v)
        edge_h = F.conv2d(b_map, self.sobel_y.to(device), padding=1)
        edge_h = torch.abs(edge_h)

        fuse_h = self.fuse_h(torch.cat((x_h, edge_h), dim=1))
        fuse_v = self.fuse_v(torch.cat((x_v, edge_v), dim=1))

        out = torch.cat((fuse_h, fuse_v), dim=1)
        out = self.conv_out(out)

        return out, b_map, edge_v, edge_h, x_h, x_v



class HybridCrossScanBlock(nn.Module):


    def __init__(self, dim, local_k=5, cross_k=21):
        super().__init__()


        self.local_conv = nn.Conv2d(
            dim, dim, kernel_size=local_k, padding=local_k // 2, groups=dim
        )

        self.h_conv = nn.Conv2d(
            dim, dim, kernel_size=(1, cross_k), padding=(0, cross_k // 2), groups=dim
        )

        self.v_conv = nn.Conv2d(
            dim, dim, kernel_size=(cross_k, 1), padding=(cross_k // 2, 0), groups=dim
        )

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.act = nn.ReLU()
        self.pw2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        input = x

        x = self.local_conv(x) + self.h_conv(x) + self.v_conv(x)


        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)

        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)

        x = x.permute(0, 3, 1, 2)

        return x + input

class GlobalExtraction(nn.Module):
  def __init__(self,dim = None):
    super().__init__()
    self.avgpool = self.globalavgchannelpool
    self.maxpool = self.globalmaxchannelpool
    self.proj = nn.Sequential(
        nn.Conv2d(2, 1, 1,1),
        nn.BatchNorm2d(1)
    )
  def globalavgchannelpool(self, x):
    x = x.mean(1, keepdim = True)
    return x

  def globalmaxchannelpool(self, x):
    x = x.max(dim = 1, keepdim=True)[0]
    return x

  def forward(self, x):
    x_ = x.clone()
    x = self.avgpool(x)
    x2 = self.maxpool(x_)

    cat = torch.cat((x,x2), dim = 1)

    proj = self.proj(cat)
    return proj

class ContextExtraction(nn.Module):
  def __init__(self, dim, reduction = None):
    super().__init__()
    self.reduction = 1 if reduction == None else 2

    self.dconv = self.DepthWiseConv2dx2(dim)
    self.proj = self.Proj(dim)

  def DepthWiseConv2dx2(self, dim):
    dconv = nn.Sequential(
        nn.Conv2d(in_channels = dim,
              out_channels = dim,
              kernel_size = 3,
              padding = 1,
              groups = dim),
        nn.BatchNorm2d(num_features = dim),
        nn.ReLU(inplace = True),
        nn.Conv2d(in_channels = dim,
              out_channels = dim,
              kernel_size = 3,
              padding = 2,
              dilation = 2),
        nn.BatchNorm2d(num_features = dim),
        nn.ReLU(inplace = True)
    )
    return dconv

  def Proj(self, dim):
    proj = nn.Sequential(
        nn.Conv2d(in_channels = dim,
              out_channels = dim //self.reduction,
              kernel_size = 1
              ),
        nn.BatchNorm2d(num_features = dim//self.reduction)
    )
    return proj
  def forward(self,x):
    x = self.dconv(x)
    x = self.proj(x)
    return x

class MultiscaleFusion(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.local= ContextExtraction(dim)
    self.global_ = GlobalExtraction()
    self.bn = nn.BatchNorm2d(num_features=dim)

  def forward(self, x, g,):
    x = self.local(x)
    g = self.global_(g)

    fuse = self.bn(x + g)
    return fuse


class MultiScaleGatedAttn(nn.Module):
    # Version 1
  def __init__(self, dim):
    super().__init__()
    self.multi = MultiscaleFusion(dim)
    self.selection = nn.Conv2d(dim, 2,1)
    self.proj = nn.Conv2d(dim, dim,1)
    self.bn = nn.BatchNorm2d(dim)
    self.bn_2 = nn.BatchNorm2d(dim)
    self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels=dim, out_channels=dim,
                  kernel_size=1, stride=1))

  def forward(self,x,g):

    x_ = x.clone()
    g_ = g.clone()



    multi = self.multi(x, g)

    multi = self.selection(multi)

    attention_weights = F.softmax(multi, dim=1)

    A, B = attention_weights.split(1, dim=1)

    x_att = A.expand_as(x_) * x_
    g_att = B.expand_as(g_) * g_

    x_att = x_att + x_
    g_att = g_att + g_


    x_sig = torch.sigmoid(x_att)
    g_att_2 = x_sig * g_att


    g_sig = torch.sigmoid(g_att)
    x_att_2 = g_sig * x_att

    interaction = x_att_2 * g_att_2

    projected = torch.sigmoid(self.bn(self.proj(interaction)))

    weighted = projected * x_

    y = self.conv_block(weighted)

    y = self.bn_2(y)
    return y

class MSDG(nn.Module):

    def __init__(self, dim=768):
        super().__init__()


        self.down1 = Downsample3(in_channels=96, out_channels=dim)
        self.down2 = Downsample2(in_channels=192, out_channels=dim)
        self.down3 = Downsample1(in_channels=384, out_channels=dim)

        self.guide_fuse = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )


        self.msga = MultiScaleGatedAttn(dim=dim)


        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x4, skip_list):

        y1 = x4.contiguous()
        y2 = self.down3(skip_list[-2].contiguous())
        y3 = self.down2(skip_list[-3].contiguous())
        y4 = self.down1(skip_list[-4].contiguous())


        g = self.guide_fuse(torch.cat([y2, y3, y4], dim=1))


        out = self.msga(y1, g)

        out = self.out_proj(out)
        return out

