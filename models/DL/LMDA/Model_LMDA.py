# https://github.com/MiaoZhengQing/LMDA-Code
import torch
import torch.nn as nn


import torch
import torch.nn as nn


class LMDA(nn.Module):
    """
    LMDA-Net for the paper
    """
    def __init__(self, chans=22, samples=1125, num_classes=4, depth=9, kernel=75, channel_depth1=24, channel_depth2=9,
                ave_depth=1, avepool=25):
        super(LMDA, self).__init__()
        self.ave_depth = ave_depth
        self.channel_weight = nn.Parameter(torch.randn(depth, 1, chans), requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)


        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, channel_depth1, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.Conv2d(channel_depth1, channel_depth1, kernel_size=(1, kernel),
                      groups=channel_depth1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.GELU(),
        )
        # self.avgPool1 = nn.AvgPool2d((1, 24))
        self.chanel_conv = nn.Sequential(
            nn.Conv2d(channel_depth1, channel_depth2, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.Conv2d(channel_depth2, channel_depth2, kernel_size=(chans, 1), groups=channel_depth2, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.GELU(),
        )

        self.norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, avepool)),
            # nn.AdaptiveAvgPool3d((9, 1, 35)),
            nn.Dropout(p=0.65),
        )

        # Define auto-padding module
        out = torch.ones((1, 1, chans, samples))
        out = torch.einsum('bdcw, hdc->bhcw', out, self.channel_weight)
        out = self.time_conv(out)
        # out = self.avgPool1(out)
        out = self.chanel_conv(out)
        out = self.norm(out)
        n_out_time = out.cpu().data.numpy().shape
        self.classifier = nn.Linear(n_out_time[-1]*n_out_time[-2]*n_out_time[-3], num_classes)

    def EEGDepthAttention(self, x):
        # x: input features with shape [N, C, H, W]

        N, C, H, W = x.size()
        # K = W if W % 2 else W + 1
        k = 7
        adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
        conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k//2, 0), bias=True).to(x.device)  # original kernel k
        softmax = nn.Softmax(dim=-2)
        x_pool = adaptive_pool(x)
        x_transpose = x_pool.transpose(-2, -3)
        y = conv(x_transpose)
        y = softmax(y)
        y = y.transpose(-2, -3)
        return y * C * x

    def forward(self, x):
        x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)  

        x_time = self.time_conv(x)  # batch, depth1, channel, samples_
        x_time = self.EEGDepthAttention(x_time)  # DA1

        x = self.chanel_conv(x_time)  # batch, depth2, 1, samples_
        x = self.norm(x)

        features = torch.flatten(x, 1)
        cls = self.classifier(features)
        return cls