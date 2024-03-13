import torch
import torch.nn as nn
import torch.nn.functional as F

# class CrossAttention(nn.Module):
#     def __init__(self, channels, reduced_channels,shape):
#         super().__init__()
#         self.query_conv = nn.Conv2d(channels, reduced_channels, 1)
#         self.key_conv = nn.Conv2d(channels, reduced_channels, 1)
#         self.value_conv = nn.Conv2d(channels, reduced_channels, 1)
#         self.final_conv = nn.Conv2d(reduced_channels, channels, 1)
#         self.norm = nn.LayerNorm([channels, shape, shape])
#
#     def forward(self, x, y):
#         # x 和 y 是输入特征图，形状为 (1, 128, 128, 128)
#         batch_size, C, width, height = x.size()
#         query = self.query_conv(x)
#         key = self.key_conv(y)
#         value = self.value_conv(y)
#
#         # 展平空间维度以便可以进行矩阵乘法
#         query = query.view(batch_size, reduced_channels, -1)
#         key = key.view(batch_size, reduced_channels, -1)
#         value = value.view(batch_size, reduced_channels, -1)
#
#         # 计算注意力得分
#         attention = F.softmax(torch.bmm(query.transpose(1, 2), key), dim=-1)
#
#         # 应用注意力得分
#         out = torch.bmm(value, attention.transpose(1, 2))
#
#         # 重塑为原始的尺寸
#         out = out.view(batch_size, reduced_channels, x.shape[2], x.shape[3])
#         out = self.final_conv(out)
#
#         # 残差连接和标准化
#         out += x
#         out = self.norm(out)
#
#         return out
class CrossAttention(nn.Module):
    def __init__(self, channels, reduced_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(channels, reduced_channels, 1)
        self.key_conv = nn.Conv2d(channels, reduced_channels, 1)
        self.value_conv = nn.Conv2d(channels, reduced_channels, 1)
        self.final_conv = nn.Conv2d(reduced_channels, channels, 1)


    def forward(self, x, y,channels,shape):
        # x 和 y 是输入特征图，形状为 (1, 128, 128, 128)
        batch_size, C, width, height = x.size()
        query = self.query_conv(x)
        key = self.key_conv(y)
        value = self.value_conv(y)
        norm = nn.LayerNorm([channels, shape, shape]).cuda()
        # 展平空间维度以便可以进行矩阵乘法
        query = query.view(batch_size, reduced_channels, -1)
        key = key.view(batch_size, reduced_channels, -1)
        value = value.view(batch_size, reduced_channels, -1)

        # 计算注意力得分
        attention = F.softmax(torch.bmm(query.transpose(1, 2), key), dim=-1)

        # 应用注意力得分
        out = torch.bmm(value, attention.transpose(1, 2))

        # 重塑为原始的尺寸
        out = out.view(batch_size, reduced_channels, x.shape[2], x.shape[3])
        out = self.final_conv(out)

        # 残差连接和标准化
        out += x
        out = norm(out)

        return out
# class CrossAttention(nn.Module):
#     def __init__(self, channels, reduced_channels):
#         super().__init__()
#         self.query_linear = nn.Linear(channels, reduced_channels)
#         self.key_linear = nn.Linear(channels, reduced_channels)
#         self.value_linear = nn.Linear(channels, reduced_channels)
#         self.final_linear = nn.Linear(reduced_channels, channels)
#
#     def forward(self, x, y,channels,shape):
#         batch_size, C, width, height = x.size()
#
#         # 展平输入
#         # x_flat = x.view(batch_size, C, -1)
#         # y_flat = y.view(batch_size, C, -1)
#         x_flat = x.permute(0, 2, 3, 1)
#         y_flat = y.permute(0, 2, 3, 1)
#         query = self.query_linear(x_flat).view(batch_size, reduced_channels, width, height)
#         key = self.key_linear(y_flat).view(batch_size, reduced_channels, width, height)
#         value = self.value_linear(y_flat).view(batch_size, reduced_channels, width, height)
#
#         norm = nn.LayerNorm([channels, shape, shape]).cuda()
#
#         # 展平空间维度以便可以进行矩阵乘法
#         query = query.view(batch_size, reduced_channels, -1)
#         key = key.view(batch_size, reduced_channels, -1)
#         value = value.view(batch_size, reduced_channels, -1)
#
#         # 计算注意力得分
#         attention = F.softmax(torch.bmm(query.transpose(1, 2), key), dim=-1)
#
#         # 应用注意力得分
#         out = torch.bmm(value, attention.transpose(1, 2))
#
#         # 重塑为原始的尺寸
#         out = out.view(batch_size, reduced_channels, x_flat.shape[1], x_flat.shape[2])
#
#         # 展平输出
#         # out_flat = out.view(batch_size, reduced_channels, -1)
#         out = self.final_linear(out.permute(0, 2, 3, 1)).view(batch_size, C, width, height)
#
#         # 残差连接和标准化
#         out += x
#         out = norm(out)
#
#         return out
# 假设 reduced_channels 是降低的通道数量
reduced_channels = 16
# model = CrossAttention(128, reduced_channels)
