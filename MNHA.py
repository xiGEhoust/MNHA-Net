from functools import partial
import math
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import RandomState
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

layer_scale = True
init_value = 1e-6


class TokenFeedForward(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DepthwiseTokenConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvFeedForward(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # print(x.shape)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

  
class DepthwiseTokenConv(nn.Module):
    def __init__(self, dim=768):
        super(DepthwiseTokenConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

    
class SparseWindowAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def partition_spatial_blocks(x, block_size):
    B,H,W,C = x.shape
    pad_h = (block_size - H % block_size) % block_size
    pad_w = (block_size - W % block_size) % block_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  
    Hp, Wp = H + pad_h, W + pad_w  
    x = x.reshape(B,Hp//block_size,block_size,Wp//block_size,block_size, C)
    x = x.permute(0,1,3,2,4,5).contiguous()
    return x, H, Hp, C

def restore_spatial_blocks(x, Ho):
    B,H,W,win_H,win_W,C = x.shape
    x = x.permute(0,1,3,2,4,5).contiguous().reshape(B,H*win_H,W*win_W, C)
    Wp = Hp = H*win_H
    Wo = Ho
    if Hp > Ho or Wp > Wo:
        x = x[:, :Ho, :Wo, :].contiguous()
    return x


def to_sparse_windows(x, sparse_size=8):
    x = x.permute(0, 2, 3, 1)
    assert x.shape[1]%sparse_size == 0 & x.shape[2]%sparse_size == 0, 'image size should be divisible by block_size'
    grid_size = x.shape[1]//sparse_size
    out, H, Hp, C = partition_spatial_blocks(x, grid_size)
    out = out.permute(0, 3, 4, 1, 2, 5).contiguous()
    out = out.reshape(-1, sparse_size, sparse_size, C)
    out = out.permute(0, 3, 1, 2)
    return out, H, Hp, C   


def from_sparse_windows(x, H, Hp, C, sparse_size=8):
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(-1, Hp//sparse_size, Hp//sparse_size, sparse_size, sparse_size, C)
    x = x.permute(0, 3, 4, 1, 2, 5).contiguous()
    out = restore_spatial_blocks(x, H)
    out = out.permute(0, 3, 1, 2)
    return out

class LocalConvBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvFeedForward(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SparseAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, sparse_size=0, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = SparseWindowAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TokenFeedForward(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        global layer_scale
        self.ls = layer_scale
        self.sparse_size = sparse_size
        if self.ls:
            global init_value
            print(f"Use layer_scale: {layer_scale}, init_values: {init_value}")
            self.gamma_1 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x_befor = x.flatten(2).transpose(1, 2)
        B, N, H, W = x.shape
        if self.ls:
            x, Ho, Hp, C = to_sparse_windows(x, self.sparse_size)
            Bf, Nf, Hf, Wf = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = self.attn(self.norm1(x))
            x = x.transpose(1, 2).reshape(Bf, Nf, Hf, Wf)
            x = from_sparse_windows(x, Ho, Hp, C, self.sparse_size)
            x = x.flatten(2).transpose(1, 2)  
            x = x_befor + self.drop_path(self.gamma_1 * x)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x), H, W))
        else:
            x, Ho, Hp, C = to_sparse_windows(x, self.sparse_size)
            Bf, Nf, Hf, Wf = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = self.attn(self.norm1(x))
            x = x.transpose(1, 2).reshape(Bf, Nf, Hf, Wf)
            x = from_sparse_windows(x, Ho, Hp, C, self.sparse_size)
            x = x.flatten(2).transpose(1, 2)
            x = x_befor + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x        



class PatchProjection(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x
    

class ChannelResponseGate(nn.Module): 
    def __init__(self, in_channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.fc = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)  
        self.relu = nn.ReLU(inplace=True)  

    def forward(self, x):
        b, c, _, _ = x.size()  
        y = self.avg_pool(x) 
        y = self.fc(y) 
        y = self.relu(y)  
        y = nn.functional.interpolate(y, size=(x.size(2), x.size(3)), mode='nearest')  
        return x * y.expand_as(x)  

class SpatialResponseGate(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False) 
        self.norm = nn.Sigmoid() 

    def forward(self, x):
        y = self.Conv1x1(x) 
        y = self.norm(y)  
        return x * y 

class DualResponseGate(nn.Module): 
    def __init__(self, in_channel):
        super().__init__()
        self.tongdao = ChannelResponseGate(in_channel)  
        self.kongjian = SpatialResponseGate(in_channel)  

    def forward(self, feature):
        spatial_response = self.kongjian(feature)  
        channel_response = self.tongdao(feature)  
        return torch.max(channel_response, spatial_response)  

class MultiDilatedFeatureAggregator(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1,
                 bn_mom=0.1):  
        super(MultiDilatedFeatureAggregator, self).__init__()
        self.branch1 = nn.Sequential(  
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential( 
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential( 
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(  
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)  
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(  
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.Hebing = DualResponseGate(in_channel=dim_out * 5)  

    def forward(self, x):
        [_, _, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        gating_map = self.Hebing(feature_cat)
        gated_feature = gating_map * feature_cat
        result = self.conv_cat(gated_feature)
        return result
def init_complex_filter(weights_real, weights_imag, criterion):
    output_chs, input_chs, num_rows, num_cols = weights_real.shape
    fan_in = input_chs
    fan_out = output_chs
    if criterion == 'glorot':
        s = 1. / np.sqrt(fan_in + fan_out) / 4.
    elif criterion == 'he':
        s = 1. / np.sqrt(fan_in) / 4.
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    rng = RandomState()
    kernel_shape = weights_real.shape
    modulus = rng.rayleigh(scale=s, size=kernel_shape)
    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
    weight_real = modulus * np.cos(phase)
    weight_imag = modulus * np.sin(phase)
    weights_real.data = torch.Tensor(weight_real)
    weights_imag.data = torch.Tensor(weight_imag)


class FourierCalibrationBlock(nn.Module):
    def __init__(self, input_chs: int, output_chs: int, num_rows: int, num_cols: int, stride=1, init='he'):
        super(FourierCalibrationBlock, self).__init__()
        self.weights_real = nn.Parameter(torch.Tensor(1, input_chs, num_rows, int(num_cols // 2 + 1)))
        self.weights_imag = nn.Parameter(torch.Tensor(1, input_chs, num_rows, int(num_cols // 2 + 1)))
        init_complex_filter(self.weights_real, self.weights_imag, init)
        self.size = (num_rows, num_cols)
        self.stride = stride

    def forward(self, x):
        x = torch.fft.rfftn(x, dim=(-2, -1), norm=None)
        x_real, x_imag = x.real, x.imag
        y_real = torch.mul(x_real, self.weights_real) - torch.mul(x_imag, self.weights_imag)
        y_imag = torch.mul(x_real, self.weights_imag) + torch.mul(x_imag, self.weights_real)
        x = torch.fft.irfftn(torch.complex(y_real, y_imag), s=self.size, dim=(-2, -1), norm=None)
        if self.stride == 2:
            x = x[..., ::2, ::2]
        return x

    def loadweight(self, ilayer):
        weight = ilayer.weight.detach().clone()
        fft_shape = self.weights_real.shape[-2]
        weight = torch.flip(weight, [-2, -1])
        pad = torch.nn.ConstantPad2d(padding=(0, fft_shape - weight.shape[-1], 0, fft_shape - weight.shape[-2]),
                                     value=0)
        weight = pad(weight)
        weight = torch.roll(weight, (-1, -1), dims=(-2, - 1))
        weight_kc = torch.fft.fftn(weight, dim=(-2, -1), norm=None).transpose(0, 1)
        weight_kc = weight_kc[..., :weight_kc.shape[-1] // 2 + 1]
        self.weights_real.data = weight_kc.real
        self.weights_imag.data = weight_kc.imag


def build_padded_conv(in_planes, out_planes, kernelsize, stride=1, dilation=1, bias=False, padding=None):
    if padding is None:
        padding = kernelsize // 2
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernelsize, stride=stride, dilation=dilation, padding=padding,
                     bias=bias)


def init_conv_weight(conv, act='linear'):
    n = conv.kernel_size[0] * conv.kernel_size[1] * conv.out_channels
    conv.weight.data.normal_(0, math.sqrt(2. / n))


def init_batchnorm_weight(m, kernelsize=3):
    n = kernelsize ** 2 * m.num_features
    m.weight.data.normal_(0, math.sqrt(2. / (n)))
    m.bias.data.zero_()


def build_activation(act):
    if act is None:
        return None
    elif act == 'relu':
        return nn.ReLU(inplace=True)
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    elif act == 'softmax':
        return nn.Softmax()
    elif act == 'linear':
        return None
    else:
        assert (False)


def build_noiseprint_net(nplanes_in, kernels, features, bns, acts, dilats, bn_momentum=0.1, padding=None):
    depth = len(features)
    assert (len(features) == len(kernels))

    layers = list()
    for i in range(0, depth):
        if i == 0:
            in_feats = nplanes_in
        else:
            in_feats = features[i - 1]

        elem = build_padded_conv(in_feats, features[i], kernelsize=kernels[i], dilation=dilats[i], padding=padding,
                                 bias=not (bns[i]))
        init_conv_weight(elem, act=acts[i])
        layers.append(elem)

        if bns[i]:
            elem = nn.BatchNorm2d(features[i], momentum=bn_momentum)
            init_batchnorm_weight(elem, kernelsize=kernels[i])
            layers.append(elem)

        elem = build_activation(acts[i])
        if elem is not None:
            layers.append(elem)

    return nn.Sequential(*layers)


class EarlyFeatureAdapter(nn.Module):

    def __init__(self, depth=3, in_channels=3, out_channels=None):
        super().__init__()
        self.depth = depth
        channels = [in_channels]
        if out_channels is None:
            out_channels = in_channels
        channels.extend([24 * 2 ** i for i in range(depth)])
        self.convs = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i + 1], 3, 1, 1),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.ReLU()
                )
                for i in range(depth)
            ]
        )
        self.final = nn.Conv2d(channels[-1], out_channels, 1, 1, 0)  
    def forward(self, x):
        x = self.convs(x)
        x = self.final(x)
        return x


class SRMHighPassFilter(nn.Module):
    def __init__(self):
        super().__init__()
        f1 = [[0, 0, 0, 0, 0],
              [0, -1, 2, -1, 0],
              [0, 2, -4, 2, 0],
              [0, -1, 2, -1, 0],
              [0, 0, 0, 0, 0]]

        f2 = [[-1, 2, -2, 2, -1],
              [2, -6, 8, -6, 2],
              [-2, 8, -12, 8, -2],
              [2, -6, 8, -6, 2],
              [-1, 2, -2, 2, -1]]

        f3 = [[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 1, -2, 1, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]]

        q = torch.tensor([[4.], [12.], [2.]]).unsqueeze(-1).unsqueeze(-1)
        filters = torch.tensor([[f1, f1, f1], [f2, f2, f2], [f3, f3, f3]], dtype=torch.float) / q
        self.register_buffer('filters', filters)
        self.truc = nn.Hardtanh(-2, 2)

    def forward(self, x):
        x = F.conv2d(x, self.filters, padding='same', stride=1)
        x = self.truc(x)
        return x


class NoiseModalMixer(nn.Module):

    def __init__(self, modals=['noiseprint', 'srm'], in_channels=[3, 3], out_channels=3):
        super().__init__()

        w = len(modals)
        assert len(modals) == len(in_channels)

        c_h = sum(in_channels)

        self.blocks = nn.ModuleList(
            [
                EarlyFeatureAdapter() for _ in range(w)
            ]
        )
        self.dropout = nn.Dropout(0.33)
        self.mixer = EarlyFeatureAdapter(in_channels=c_h, out_channels=out_channels)

    def forward(self, x):
        m = []
        for m_i, blk in enumerate(self.blocks):
            m.append(blk(x[m_i]))

        x = torch.cat(m, dim=1)
        x = self.dropout(x)
        x = self.mixer(x)
        return x


class NoiseModalitiesExtractor(nn.Module):
    def __init__(self,
                 modals: list = ('noiseprint','srm'),
                 noiseprint_path: str = None):
        super().__init__()
        self.mod_extract = []
        if 'noiseprint' in modals:
            num_levels = 17
            out_channel = 1
            self.noiseprint = build_noiseprint_net(3, kernels=[3, ] * num_levels,
                                       features=[64, ] * (num_levels - 1) + [out_channel],
                                       bns=[False, ] + [True, ] * (num_levels - 2) + [False, ],
                                       acts=['relu', ] * (num_levels - 1) + ['linear', ],
                                       dilats=[1, ] * num_levels,
                                       bn_momentum=0.1, padding=1)

            if noiseprint_path:
                np_weights = noiseprint_path
                assert os.path.isfile(np_weights)
                dat = torch.load(np_weights, map_location=torch.device('cpu'), weights_only=True)
                print(f'Noiseprint++ weights: {np_weights}')
                if isinstance(dat, dict) and 'network' in dat:
                    dat = dat['network']
                self.noiseprint.load_state_dict(dat)

            self.noiseprint.eval()
            for param in self.noiseprint.parameters():
                param.requires_grad = False
            self.mod_extract.append(self.noiseprint)

        if 'srm' in modals:
            self.srm = SRMHighPassFilter()
            self.mod_extract.append(self.srm)

        self.dropout = nn.Dropout(0.33)
        self.FF = FourierCalibrationBlock(1, 3, 512, 512)

    def forward(self, x) -> list:
        #out = []
        y1=self.FF(x)
        for modal in self.mod_extract:
            y = modal(y1)
            if y.size()[-3] == 1:
                y = y.repeat((1, 3, 1, 1))
            #out.append(y)
        #y = torch.cat(out, dim=1)
        y = self.dropout(y)
        return y
        
class NoiseChannelReducer(nn.Module):
    def __init__(self):
        super(NoiseChannelReducer,self).__init__()
        self.c1=nn.Conv2d(3,320,1)
        self.n1 = nn.BatchNorm2d(320, momentum=0.1)
        self.p1=nn.AvgPool2d(16,stride=16)
        self.d1 = nn.Dropout()

    def forward(self,x):
        x=self.c1(x)
        x=self.n1(x)
        x=self.p1(x)
        x=self.d1(x)
        return x
class MNHA(nn.Module):
    def __init__(self, layers=[5, 8, 20, 7], img_size=224, in_chans=3, s_blocks3=[ 8, 4, 2, 1], s_blocks4=[2, 1], embed_dim=[64, 128, 320, 512],
                 head_dim=64, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=None, pretrained_path=None,
                 ):
        super().__init__()
        self.pretrained_path=pretrained_path
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6) 
        self.M1=MultiDilatedFeatureAggregator(dim_in=320,dim_out=320)
        self.M2=MultiDilatedFeatureAggregator(dim_in=512,dim_out=512)
        
        self.patch_embed1 = PatchProjection(
            img_size=img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchProjection(
            img_size=img_size // 4, patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchProjection(
            img_size=img_size // 8, patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2])
        self.patch_embed4 = PatchProjection(
            img_size=img_size // 16, patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3])

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]  # stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dim]
        self.blocks1 = nn.ModuleList([
            LocalConvBlock(
                dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(layers[0])])
        self.norm1=norm_layer(embed_dim[0])
        self.blocks2 = nn.ModuleList([
            LocalConvBlock(
                dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+layers[0]], norm_layer=norm_layer)
            for i in range(layers[1])])
        self.norm2 = norm_layer(embed_dim[1])
       
        self.blocks3 = nn.ModuleList()
        for i in range(layers[2]):
            block =  SparseAttentionBlock(
                            dim=embed_dim[2], num_heads=num_heads[2], sparse_size=32//s_blocks3[i//5], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+layers[0]+layers[1]], norm_layer=norm_layer)
            self.blocks3.append(block)
        self.norm3 = norm_layer(embed_dim[2])
        self.blocks4 = nn.ModuleList()
        for i in range(layers[3]):
            block = SparseAttentionBlock(
                            dim=embed_dim[3], num_heads=num_heads[3], sparse_size=16//s_blocks4[i//4], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+layers[0]+layers[1]+layers[2]], norm_layer=norm_layer)
            self.blocks4.append(block)
        self.norm4 = norm_layer(embed_dim[3])
        
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()
            
        self._uniformer_init_weights()
        self.apply(self._init_weights)
        self.c1=NoiseChannelReducer()
        self.modal_ext = NoiseModalitiesExtractor(['noiseprint','srm'], "./checkpoint/noiseprint++.pth")
        
       
    def forward(self, x):
        outputs = self.forward_features(x)
        return outputs

    def forward_features(self, x):
        outputs = {}

        xs = self.modal_ext(x)
        xs = self.c1(xs)
        outputs.update({"third" + str(4 // 4): xs})

        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        x = self.patch_embed3(x)
        x1 = self.M1(x)
        for index, blk in enumerate(self.blocks3):
            x = blk(x)
            if (index+1)%5==0 and (index+1)//5!=4 and (index+1)//5!=1:
                outputs.update({"third"+str((index+1)//5): x})
        x = x1 + x
        x_out = self.norm3(x.permute(0, 2, 3, 1))
        outputs.update({"third": x_out.permute(0, 3, 1, 2).contiguous()})
        x = self.patch_embed4(x)
        x2 = self.M2(x)
        for index, blk in enumerate(self.blocks4):
            x = blk(x)
            if (index+1)%4==0:
                outputs.update({"last"+str((index+1)//4): x})
        x = x2 + x
        x_out = self.norm4(x.permute(0, 2, 3, 1))
        outputs.update({"last": x_out.permute(0, 3, 1, 2).contiguous()})
        return outputs

    def _uniformer_init_weights(self):
        if self.pretrained_path != None:
            state_dict = torch.load(self.pretrained_path, map_location='cpu')
            new_state_dict = {}
            for k, v in state_dict['model'].items():
                if k.startswith('backbone.'):
                    new_key = k[len('backbone.'):]
                else:
                    new_key = k
                new_state_dict[new_key] = v  
            self.load_state_dict(new_state_dict, strict=False)
            print('load pretrained weights from \'{}\'.'.format(self.pretrained_path))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



