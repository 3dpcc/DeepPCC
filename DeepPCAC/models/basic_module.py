import torch
import MinkowskiEngine as ME
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import numpy as np

import math
import pytorch3d.ops


# class Encoder(torch.nn.Module):
#   def __init__(self, channels=128):
#     super().__init__()
#     self.conv0 = ME.MinkowskiConvolution(
#         in_channels=3,
#         out_channels=64,
#         kernel_size=3,
#         stride=1,
#         bias=True,
#         dimension=3)
#     self.conv0_0 = ME.MinkowskiConvolution(
#         in_channels=64,
#         out_channels=channels,
#         kernel_size=3,
#         stride=2,
#         bias=True,
#         dimension=3)

#     self.conv1 = ME.MinkowskiConvolution(
#         in_channels=channels,
#         out_channels=channels,
#         kernel_size=3,
#         stride=1,
#         bias=True,
#         dimension=3)
#     self.conv1_0 = ME.MinkowskiConvolution(
#         in_channels=channels,
#         out_channels=channels,
#         kernel_size=3,
#         stride=2,
#         bias=True,
#         dimension=3)

#     self.conv2 = ME.MinkowskiConvolution(
#         in_channels=channels,
#         out_channels=channels,
#         kernel_size=3,
#         stride=1,
#         bias=True,
#         dimension=3)
#     self.conv2_0 = ME.MinkowskiConvolution(
#         in_channels=channels,
#         out_channels=channels,
#         kernel_size=3,
#         stride=2,
#         bias=True,
#         dimension=3)

#     self.relu = ME.MinkowskiReLU(inplace=True)

#   def forward(self, x):
#     out = self.relu(self.conv0_0(self.conv0(x)))
#     out = self.relu(self.conv1_0(self.conv1(out)))
#     out = self.conv2_0(self.conv2(out))

#     return out


# class Decoder(torch.nn.Module):
#   def __init__(self, channels=128):
#     super().__init__()
#     self.deconv0 = ME.MinkowskiConvolutionTranspose(
#         in_channels=channels,
#         out_channels=channels,
#         kernel_size=3,
#         stride=2,
#         bias=True,
#         dimension=3)
#     self.deconv0_0 = ME.MinkowskiConvolutionTranspose(
#         in_channels=channels,
#         out_channels=channels,
#         kernel_size=3,
#         stride=1,
#         bias=True,
#         dimension=3)

#     self.deconv1 = ME.MinkowskiConvolutionTranspose(
#         in_channels=channels,
#         out_channels=channels,
#         kernel_size=3,
#         stride=2,
#         bias=True,
#         dimension=3)
#     self.deconv1_0 = ME.MinkowskiConvolutionTranspose(
#         in_channels=channels,
#         out_channels=channels,
#         kernel_size=3,
#         stride=1,
#         bias=True,
#         dimension=3)

#     self.deconv2 = ME.MinkowskiConvolutionTranspose(
#         in_channels=channels,
#         out_channels=64,
#         kernel_size=3,
#         stride=2,
#         bias=True,
#         dimension=3)
#     self.deconv2_0 = ME.MinkowskiConvolutionTranspose(
#         in_channels=64,
#         out_channels=3,
#         kernel_size= 3,
#         stride=1,
#         bias=True,
#         dimension=3)

#     self.relu = ME.MinkowskiReLU(inplace=True)

#   def forward(self, x):
#     out = self.relu(self.deconv0_0(self.deconv0(x)))
#     out = self.relu(self.deconv1_0(self.deconv1(out)))
#     out = self.deconv2_0(self.deconv2(out))

#     return out

class MLP_block(torch.nn.Module):
    def __init__(self, channels=192):
        super().__init__()
        self.linear0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=True,
            dimension=3
        )
        self.linear1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=128,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=True,
            dimension=3
        )
        self.linear2 = ME.MinkowskiConvolution(
            in_channels=128,
            out_channels=64,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=True,
            dimension=3
        )
        self.linear3 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=9,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=True,
            dimension=3
        )
        self.relu = ME.MinkowskiReLU(inplace=True)
    def forward(self,x):
        return self.linear3(self.relu(self.linear2(self.relu(self.linear1(self.relu(self.linear0(x)))))))




class Transformer_block(torch.nn.Module):
    def __init__(self, channels, head, k):
        super(Transformer_block, self).__init__()

        self.layer_norm_1 = nn.LayerNorm(channels)
        self.linear = torch.nn.Linear(channels, channels)
        self.layer_norm_2 = nn.LayerNorm(channels)

        self.sa = SA_Layer(channels, head, k)

    def forward(self, x, knn_feature, knn_xyz):
        x1 = x + self.sa(x, knn_feature, knn_xyz)
        x1_F = x1.F

        x1_F = self.layer_norm_1(x1_F)
        x1_F = x1_F + self.linear(x1_F)
        x1_F = self.layer_norm_2(x1_F)

        x1 = ME.SparseTensor(features=x1_F, coordinate_map_key=x1.coordinate_map_key,
                             coordinate_manager=x1.coordinate_manager)

        return x1


class Point_Transformer_Last(torch.nn.Module):
    def __init__(self, block=2, channels=128, head=1, k=16):
        super(Point_Transformer_Last, self).__init__()
        self.head = head
        self.k = k
        self.layers = torch.nn.ModuleList()
        for i in range(block):
            self.layers.append(Transformer_block(channels, head, k))

    def forward(self, x):
        out = x
        x_C = out.C.unsqueeze(0).float()
        dist, idx, _ = pytorch3d.ops.knn_points(x_C, x_C, K=self.k)
        knn_xyz =  pytorch3d.ops.knn_gather(x_C[:,:,1:], idx)
        center_xyz = x_C[:, :, 1:].unsqueeze(2)

        knn_xyz_norm = knn_xyz - center_xyz
        knn_xyz_norm = knn_xyz_norm.squeeze(0)
        knn_xyz_norm = knn_xyz_norm / knn_xyz_norm.max()

        for transformer in self.layers:
            out_F = out.F.unsqueeze(0).float()
            knn_feature = pytorch3d.ops.knn_gather(out_F[:,:,:], idx).squeeze(0)
            out = transformer(x, knn_feature, knn_xyz_norm)

        return out


class SA_Layer(nn.Module):
    def __init__(self, channels, head=1, k=16):
        super(SA_Layer, self).__init__()
        self.channels = channels
        self.q_conv = torch.nn.Linear(channels, channels)
        self.k_conv = torch.nn.Linear(channels + 3, channels)
        self.v_conv = torch.nn.Linear(channels + 3, channels)
        self.d = math.sqrt(channels)
        self.head = head
        self.k = k

    def forward(self, x, knn_feature, knn_xyz):
        x_q = x.F

        new_knn_feature = torch.cat((knn_feature, knn_xyz), dim=2)

        Q = self.q_conv(x_q).view(-1, self.head, self.channels // self.head)
        K = self.k_conv(new_knn_feature).view(-1, self.head, self.k, self.channels // self.head)
        attention_map = torch.einsum('nhd,nhkd->nhk', Q, K)
        attention_map = F.softmax(attention_map / self.d, dim=-1)
        # print(attention_map)

        V = self.v_conv(new_knn_feature).view(-1, self.head, self.k, self.channels // self.head)
        attention_feature = torch.einsum('nhk,nhkd->nhd', attention_map, V)
        attention_feature = attention_feature.view(-1, self.channels)

        new_x = ME.SparseTensor(features=attention_feature, coordinate_map_key=x.coordinate_map_key,
                                coordinate_manager=x.coordinate_manager)

        return new_x


class ResNet(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv0(x))
        out = self.conv1(out)
        out += x

        return out


# def make_layer(block, block_layers, channels):
#     layers = []
#     for i in range(block_layers):
#         layers.append(block(channels=channels))
#
#     return torch.nn.Sequential(*layers)

class InceptionResNet(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels // 4,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)
        self.conv0_1 = ME.MinkowskiConvolution(
            in_channels=channels // 4,
            out_channels=channels // 4,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv0_2 = ME.MinkowskiConvolution(
            in_channels=channels // 4,
            out_channels=channels // 2,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels // 4,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1_1 = ME.MinkowskiConvolution(
            in_channels=channels // 4,
            out_channels=channels // 2,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out0 = self.conv0_2(self.relu(self.conv0_1(self.relu(self.conv0_0(x)))))
        out1 = self.conv1_1(self.relu(self.conv1_0(x)))
        out = ME.cat(out0, out1)
        return out + x


def make_layer(block, block_layers, channels):
    """make stacked InceptionResNet layers.
    """
    layers = []
    for i in range(block_layers):
        layers.append(block(channels=channels))

    return torch.nn.Sequential(*layers)

class Transfomer_Based_Module(torch.nn.Module):
    def __init__(self, channels=128):
        super(Transfomer_Based_Module, self).__init__()
        self.conv0_0 = ME.MinkowskiConvolution(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv_down0 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.knn0 = Point_Transformer_Last(block=2, channels=channels, head=1, k=16)

        self.conv1_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv_down1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.knn1 = Point_Transformer_Last(block=2, channels=channels, head=1, k=16)

        self.conv2_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv_down2 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.knn2 = Point_Transformer_Last(block=2, channels=channels, head=1, k=16)
        self.relu = ME.MinkowskiReLU(inplace=True)
    def forward(self, x):
        out = self.relu(self.knn0(self.conv_down0(self.conv0_0(x))))
        out = self.relu(self.knn1(self.conv_down1(self.conv1_0(out))))
        out = self.knn2(self.conv_down2(self.conv2_0(out)))
        return out

class Transfomer_Decoder(torch.nn.Module):
    def __init__(self, channels=128):
        super(Transfomer_Decoder, self).__init__()
        self.up0 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.deconv0_1 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.knn0 = Point_Transformer_Last(block=2, channels=channels, head=1, k=16)

        self.up1 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.deconv1_1 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.knn1 = Point_Transformer_Last(block=2, channels=channels, head=1, k=16)

        self.up2 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=64,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)

        self.deconv2_1 = ME.MinkowskiConvolutionTranspose(
            in_channels=64,
            out_channels=3,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.knn2 = Point_Transformer_Last(block=2, channels=channels, head=1, k=16)
        self.relu = ME.MinkowskiReLU(inplace=True)
    def forward(self, x):
        out = self.relu(self.deconv0_1(self.up0(self.knn0(x))))
        out = self.relu(self.deconv1_1(self.up1(self.knn1(out))))
        out = self.deconv2_1(self.up2(self.knn2(out)))
        return out

class Encoder(torch.nn.Module):
    def __init__(self, channels=128, is_deep=False):
        super().__init__()
        self.conv0_0 = ME.MinkowskiConvolution(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.IRN0_0 = InceptionResNet(channels=64)
        self.conv_down0 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        # self.knn0 = Point_Transformer_Last(block=2, channels=channels, head=1, k=16)

        if is_deep: self.block0 = make_layer(
            block=InceptionResNet,
            block_layers=3,
            channels=channels)

        self.conv1_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.IRN1_0 = InceptionResNet(channels=channels)
        self.conv_down1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        # self.knn1 = Point_Transformer_Last(block=2, channels=channels, head=1, k=16)
        if is_deep: self.block1 = make_layer(
            block=ResNet,
            block_layers=3,
            channels=channels)

        self.conv2_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.IRN2_0 = InceptionResNet(channels=channels)
        self.conv_down2 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        # self.knn2 = Point_Transformer_Last(block=2, channels=channels, head=1, k=16)

        self.avg0 = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)
        self.avg1 = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)
        self.avg2 = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)

        self.fusion0 = ME.MinkowskiConvolution(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.fusion1 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.fusion2 = ME.MinkowskiConvolution(
            in_channels=channels * 3,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        #
        self.transfomer = Transfomer_Based_Module(channels=128)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv_down0(self.IRN0_0(self.relu(self.conv0_0(x)))))
        if 'block0' in self._modules: out = self.block0(out)
        out = self.relu(self.conv_down1(self.IRN1_0(self.relu(self.conv1_0(out)))))
        if 'block1' in self._modules: out = self.block1(out)
        out = self.conv_down2(self.IRN2_0(self.relu(self.conv2_0(out))))

        out_downsample = self.avg2(self.avg1(self.avg0(x)))
        out_feats = self.relu(self.fusion1(self.fusion0(out_downsample)))
        out_transfomer = self.transfomer(x)
        out = self.fusion2(ME.cat(out_feats, out, out_transfomer))

        return out


class Decoder(torch.nn.Module):
    def __init__(self, channels=128, is_deep=False):
        super().__init__()
        self.channels = channels
        self.up0 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.decIRN0_1 = InceptionResNet(channels=channels)
        self.deconv0_1 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)


        self.up1 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.decIRN1_1 = InceptionResNet(channels=channels)
        self.deconv1_1 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.up2 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=64,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)

        self.decIRN2_1 = InceptionResNet(channels=64)
        self.deconv2_1 = ME.MinkowskiConvolutionTranspose(
            in_channels=64,
            out_channels=3,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.defusion = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=channels * 2,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.transformer_deocer = Transfomer_Decoder(channels=channels)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.softmax = ME.MinkowskiSoftmax()
    def forward(self, x):
        out = self.defusion(x)

        x_1 = ME.SparseTensor(features=out.F[:,:self.channels], coordinate_manager=out.coordinate_manager,
                              coordinate_map_key=out.coordinate_map_key, device=out.device)
        x_2 = ME.SparseTensor(features=out.F[:,self.channels:], coordinate_manager=out.coordinate_manager,
                              coordinate_map_key=out.coordinate_map_key, device=out.device)
        out = self.relu(self.deconv0_1(self.decIRN0_1(self.relu(self.up0(x_1)))))
        out = self.relu(self.deconv1_1(self.decIRN1_1(self.relu(self.up1(out)))))
        out_1 = self.deconv2_1(self.decIRN2_1(self.relu(self.up2(out))))

        out_2 = self.softmax(self.transformer_deocer(x_2))

        #out = out_1 + out_2
       
        out = ME.SparseTensor(features=(out_1.F * out_2.F), coordinate_manager=out_1.coordinate_manager,
                             coordinate_map_key=out_1.coordinate_map_key, device=out_1.device)

        return out


class HyperEncoder(torch.nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.conv_in = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv0_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv_in(x))
        out = self.relu(self.conv0_0(self.conv0(x)))
        out = self.conv1_0(self.conv1(out))

        return out


class HyperDecoder(torch.nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.deconv0 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.deconv0_0 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.deconv1 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=channels * 2,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.deconv1_0 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels * 2,
            out_channels=channels * 2,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.deconv_out = ME.MinkowskiConvolutionTranspose(
            in_channels=channels * 2,
            out_channels=channels * 2,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.deconv0_0(self.deconv0(x)))
        out = self.relu(self.deconv1_0(self.deconv1(out)))
        out = self.deconv_out(out)

        return out


###########################################################################
class MaskSparseCNN(ME.MinkowskiConvolution):
    def __init__(self, in_channels, out_channels, kernel_size=-1, stride=1,
                 dilation=1, bias=False, dimension=None):
        super(MaskSparseCNN, self).__init__(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            dilation=dilation,
                                            bias=bias,
                                            dimension=dimension)
        n_kernel, _, _ = self.kernel.size()
        mask = torch.zeros(self.kernel.size())
        mask[:n_kernel // 2, :, :] = 1
        mask[n_kernel // 2:, :, :] = 0
        self.register_buffer('mask', mask)

    def forward(self, x):
        self.kernel.data *= self.mask

        return super(MaskSparseCNN, self).forward(x)


class ContextModelBase(torch.nn.Module):
    def __init__(self, channels=128):
        super(ContextModelBase, self).__init__()
        self.channels = channels
        self.maskedconv = MaskSparseCNN(in_channels=channels,
                                        out_channels=channels * 2,
                                        kernel_size=5,
                                        stride=1,
                                        dilation=1,
                                        bias=True,
                                        dimension=3)
        self.conv0 = ME.MinkowskiConvolution(in_channels=channels * 2,
                                             out_channels=channels * 2,
                                             kernel_size=1,
                                             stride=1,
                                             bias=True,
                                             dimension=3)
        self.conv1 = ME.MinkowskiConvolution(in_channels=channels * 2,
                                             out_channels=channels * 2,
                                             kernel_size=1,
                                             stride=1,
                                             bias=True,
                                             dimension=3)
        self.conv2 = ME.MinkowskiConvolution(in_channels=channels * 2,
                                             out_channels=channels * 2,
                                             kernel_size=1,
                                             stride=1,
                                             bias=True,
                                             dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        context = self.maskedconv(x)
        out = self.relu(self.conv0(context))
        out = self.relu(self.conv1(out))
        out = self.conv2(out)
        params = out.F
        loc = params[:, :self.channels]
        scale = params[:, self.channels:]

        return loc, scale.abs()


class ContextModelHyper(torch.nn.Module):
    def __init__(self, channels=128):
        super(ContextModelHyper, self).__init__()
        self.channels = channels
        self.maskedconv = MaskSparseCNN(in_channels=channels,
                                        out_channels=channels * 2,
                                        kernel_size=5,
                                        stride=1,
                                        dilation=1,
                                        bias=True,
                                        dimension=3)
        self.conv0 = ME.MinkowskiConvolution(in_channels=channels * 4,
                                             out_channels=channels * 3,
                                             kernel_size=1,
                                             stride=1,
                                             bias=True,
                                             dimension=3)
        self.conv1 = ME.MinkowskiConvolution(in_channels=channels * 3,
                                             out_channels=channels * 2,
                                             kernel_size=1,
                                             stride=1,
                                             bias=True,
                                             dimension=3)
        self.conv2 = ME.MinkowskiConvolution(in_channels=channels * 2,
                                             out_channels=channels * 2,
                                             kernel_size=1,
                                             stride=1,
                                             bias=True,
                                             dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x, hyper):
        context = self.maskedconv(x)
        if context.coordinate_manager == hyper.coordinate_manager:
            context_hyper = ME.cat(context, hyper)
        else:
            context_hyper = ME.SparseTensor(
                features=torch.cat((context.F, hyper.F), dim=-1),
                coordinate_map_key=context.coordinate_map_key,
                coordinate_manager=context.coordinate_manager,
                device=context.device)
        out = self.relu(self.conv0(context_hyper))
        out = self.relu(self.conv1(out))
        out = self.conv2(out)
        params = out.F
        loc = params[:, :self.channels]
        scale = params[:, self.channels:]

        return loc, scale.abs()


class Enhancer(torch.nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        # self.block0 = make_layer(block=ResNet, block_layers=3, channels=32)
        self.res0 = InceptionResNet(channels=64)
        # self.knn0 = Point_Transformer_Last(block=4, channels=64, head=1, k=16)

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        # self.block1 = make_layer(block=ResNet, block_layers=3, channels=64)
        self.res1 = InceptionResNet(channels=128)
        # self.knn1 = Point_Transformer_Last(channels=128, head=1, k=16)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        # self.block2 = make_layer(block=ResNet, block_layers=3, channels=64)
        self.res2 = InsertResNet(channels=128)
        # self.knn2 = Point_Transformer_Last(channels=128, head=1, k=16)

        self.conv3 = ME.MinkowskiConvolution(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3
        )
        # self.block3 = make_layer(block=ResNet, block_layers=3, channels=32)
        self.res3 = InceptionResNet(channels=64)
        # self.knn3 = Point_Transformer_Last(channels=128, head=1, k=16)

        # self.conv4 = ME.MinkowskiConvolution(
        #     in_channels=64,
        #     out_channels=64,
        #     kernel_size=3,
        #     stride=1,
        #     bias=True,
        #     dimension=3
        # )
        # self.block3 = make_layer(block=ResNet, block_layers=3, channels=32)
        # self.res4 = InsertResNet(channels=64)

        self.conv_out0 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=3,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.conv_out1 = ME.MinkowskiConvolution(
            in_channels=1,
            out_channels=3,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.res0(self.relu(self.conv0(x)))
        # out = self.knn0(out)

        out = self.res1(self.relu(self.conv1(out)))
        # out = self.knn1(out)

        out = self.res2(self.relu(self.conv2(out)))
        # out = self.knn2(out)

        out = self.res3(self.relu(self.conv3(out)))

        # out = self.res4(self.relu(self.conv4(out)))

        out = self.conv_out0(out)
        out = out + self.conv_out1(x)

        return out


class Mutiscale_enhancer(torch.nn.Module):
    def __init__(self, channels = 128):
        super().__init__()
        self.enhancer0 = Enhancer(channels=3)

        self.down1 = ME.MinkowskiConvolution(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)

        self.enhancer1 = Enhancer(channels=32)

        self.upsamp1 = ME.MinkowskiConvolutionTranspose(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3
        )

        self.MlP = MLP_block(channels=128)
        self.conv_outx = ME.MinkowskiConvolution(
            in_channels=3,
            out_channels=9,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)
    def forward(self, x):
        out0 = self.enhancer0(x)

        out1 = self.upsamp1(self.enhancer1(self.down1(x)))

        out = ME.cat(out0, out1)

        return self.MlP(out) + self.conv_outx(x)


# class Enhancer(torch.nn.Module):


#     def __init__(self, channels=128):
#         super().__init__()
#         self.conv0 = ME.MinkowskiConvolution(
#             in_channels=3,
#             out_channels=channels,
#             kernel_size=3,
#             stride=1,
#             bias=True,
#             dimension=3)
#         self.block0 = make_layer(
#             block=ResNet,
#             block_layers=3,
#             channels=channels)
#         self.conv1 = ME.MinkowskiConvolution(
#             in_channels=channels,
#             out_channels=9,
#             kernel_size=3,
#             stride=1,
#             bias=True,
#             dimension=3)
#         self.conv2 = ME.MinkowskiConvolution(
#             in_channels=3,
#             out_channels=9,
#             kernel_size=3,
#             stride=1,
#             bias=True,
#             dimension=3)
#         self.relu = ME.MinkowskiReLU(inplace=True)
#
#     def forward(self, x):
#         out = self.relu(self.conv0(x))
#         out = self.block0(out)
#         out = self.conv1(out)
#         out = out + self.conv2(x)
#
#         return out


if __name__ == '__main__':
    # encoder = Encoder(128, 3)
    # print(encoder)
    # decoder = Decoder(128, 3)
    # print(decoder)
    #
    # hyperEncoder = HyperEncoder(128)
    # print(hyperEncoder)
    # hyperDecoder = HyperDecoder(128)
    # print(hyperDecoder)
    #
    # contextModelBase = ContextModelBase(128)
    # print(contextModelBase)
    #
    # contextModelHyper = ContextModelHyper(128)
    # print(contextModelHyper)
    enhance = Encoder()
    print(enhance)
    print('params:', sum(param.numel() for param in enhance.parameters()))
