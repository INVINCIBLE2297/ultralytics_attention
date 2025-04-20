import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
 
__all__ = (
 "CoordAtt",
 "SKAttention",
 "CBAM_Authentic",
 "CBAM",
 "ECAAttention",
 "GAM",
)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
 
    def forward(self, x):
        return self.relu(x + 3) / 6
 
 
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
 
    def forward(self, x):
        return x * self.sigmoid(x)
 
 
class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
 
        mip = max(8, inp // reduction)
 
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
 
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
 
    def forward(self, x):
        identity = x
 
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
 
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
 
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
 
        out = identity * a_w * a_h
 
        return out

class SKAttention(nn.Module):
    def __init__(self, channel=512, kernels=[1,3,5,7],reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel//reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k//2, groups=group)),
                    ('bn', nn.BatchNorm2d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        # split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w

        # fuse
        U = sum(conv_outs)  # bs,c,h,w

        # reduction channel
        S = U.mean(-1).mean(-1)  # bs,c
        Z=self.fc(S)  # bs,d

        # calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        attention_weights=torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weights=self.softmax(attention_weights)  # k,bs,channel,1,1

        # fuse
        V=(attention_weights*feats).sum(0)
        return V

  
class ChannelAttentionAuthentic(nn.Module):
    """论文标准的通道注意力模块 (含双池化+共享MLP)"""
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化分支
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化分支

        # 共享的MLP (使用1x1卷积实现)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),  # 降维
            nn.ReLU(inplace=True),                                                        # 激活
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False)   # 升维
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 双池化分支处理
        avg_out = self.mlp(self.avg_pool(x))  # [B, C, 1, 1]
        max_out = self.mlp(self.max_pool(x))  # [B, C, 1, 1]

        # 逐元素相加 + Sigmoid
        channel_att = self.sigmoid(avg_out + max_out)  # [B, C, 1, 1]

        # 应用注意力权重
        return x * channel_att  # 广播乘法 [B, C, H, W]


class SpatialAttentionAuthentic(nn.Module):
    """论文标准的空间注意力模块"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in {3, 7}, "kernel_size必须是3或7"
        padding = 3 if kernel_size == 7 else 1

        # 通道维度池化 + 7x7卷积
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通道维度池化 (平均和最大)
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]

        # 拼接 + 卷积
        x_cat = torch.cat([avg_out, max_out], dim=1)    # [B, 2, H, W]
        spatial_att = self.sigmoid(self.conv(x_cat))    # [B, 1, H, W]

        # 应用注意力权重
        return x * spatial_att  # 广播乘法 [B, C, H, W]


class CBAM_Authentic(nn.Module):
    """完整的CBAM模块 (通道在前 + 空间在后)"""
    def __init__(self, channels: int, reduction_ratio: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_att = ChannelAttentionAuthentic(channels, reduction_ratio)
        self.spatial_att = SpatialAttentionAuthentic(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 顺序执行：通道注意力 -> 空间注意力
        x = self.channel_att(x)    # 通道细化
        x = self.spatial_att(x)    # 空间细化
        return x


"""
通道注意力模型: 通道维度不变，压缩空间维度。该模块关注输入图片中有意义的信息。
1）假设输入的数据大小是(b,c,w,h)
2）通过自适应平均池化使得输出的大小变为(b,c,1,1)
3）通过2d卷积和sigmod激活函数后，大小是(b,c,1,1)
4）将上一步输出的结果和输入的数据相乘，输出数据大小是(b,c,w,h)。
"""


class ChannelAttention(nn.Module):
    # Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))


"""
空间注意力模块：空间维度不变，压缩通道维度。该模块关注的是目标的位置信息。
1） 假设输入的数据x是(b,c,w,h)，并进行两路处理。
2）其中一路在通道维度上进行求平均值，得到的大小是(b,1,w,h)；另外一路也在通道维度上进行求最大值，得到的大小是(b,1,w,h)。
3） 然后对上述步骤的两路输出进行连接，输出的大小是(b,2,w,h)
4）经过一个二维卷积网络，把输出通道变为1，输出大小是(b,1,w,h)
4）将上一步输出的结果和输入的数据x相乘，最终输出数据大小是(b,c,w,h)。
"""


class SpatialAttention(nn.Module):
    # Spatial-attention module
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    # Convolutional Block Attention Module
    def __init__(self, c1, kernel_size=7):  # ch_in, kernels
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))



class ECAAttention(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
 
    def __init__(self, c1, k_size=3):
        super(ECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)

 
class GAM(nn.Module):
    def __init__(self, in_channels, rate=4):
        super().__init__()
        out_channels = in_channels
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        inchannel_rate = int(in_channels / rate)
 
        self.linear1 = nn.Linear(in_channels, inchannel_rate)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(inchannel_rate, in_channels)
 
        self.conv1 = nn.Conv2d(in_channels, inchannel_rate, kernel_size=7, padding=3, padding_mode='replicate')
 
        self.conv2 = nn.Conv2d(inchannel_rate, out_channels, kernel_size=7, padding=3, padding_mode='replicate')
 
        self.norm1 = nn.BatchNorm2d(inchannel_rate)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        b, c, h, w = x.shape
        # B,C,H,W ==> B,H*W,C
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
 
        # B,H*W,C ==> B,H,W,C
        x_att_permute = self.linear2(self.relu(self.linear1(x_permute))).view(b, h, w, c)
 
        # B,H,W,C ==> B,C,H,W
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
 
        x = x * x_channel_att
 
        x_spatial_att = self.relu(self.norm1(self.conv1(x)))
        x_spatial_att = self.sigmoid(self.norm2(self.conv2(x_spatial_att)))
 
        out = x * x_spatial_att
 
        return out
