import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Optional, List

def CalculateOutSize(blocks: List[nn.Module], Chans: int, Samples: int) -> int:
    """
    Compute output feature size of conv block sequence
    
    Args:
        blocks: list of conv blocks
        Chans: number of input channels
        Samples: number of time samples
        
    Returns:
        time dimension of output features
    """
    # Create dummy input tensor (batch_size=1, channels=1, Chans, Samples)
    x = torch.zeros((1, 1, Chans, Samples))
    
    # Run through all conv blocks
    for block in blocks:
        x = block(x)
    
    # Return time dimension size
    return x.size(-1)


class DeepConvNet(nn.Module):
    def __init__(self,
                 n_classes: int,
                 Chans: int,
                 Samples: int,
                 dropoutRate: Optional[float] = 0.5,
                 bn_track=True,
                 TemporalKernel_Times=1):
        super(DeepConvNet, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        self.bn_track = bn_track  # track_running_state of BatchNorm2d, if fedbs, bn_track = False
        self.TemporalKernel_Times = TemporalKernel_Times  # multiplier for temporal convolution kernel size compared to original

        self.block1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 10 * self.TemporalKernel_Times))),
            ('conv2', nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(Chans, 1))),
            ('bn', nn.BatchNorm2d(num_features=25, track_running_stats=self.bn_track)), ('elu', nn.ELU()),
            ('maxpool', nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))),
            ('dropout', nn.Dropout(self.dropoutRate))]))

        self.block2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1, 10 * self.TemporalKernel_Times))),
            ('bn', nn.BatchNorm2d(num_features=50, track_running_stats=self.bn_track)), ('elu', nn.ELU()),
            ('maxpool', nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))),
            ('dropout', nn.Dropout(self.dropoutRate))]))

        self.block3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1, 10 * self.TemporalKernel_Times))),
            ('bn', nn.BatchNorm2d(num_features=100, track_running_stats=self.bn_track)), ('elu', nn.ELU()),
            ('maxpool', nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))),
            ('dropout', nn.Dropout(self.dropoutRate))]))

        self.block4 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(1, 10 * self.TemporalKernel_Times))),
            ('bn', nn.BatchNorm2d(num_features=200, track_running_stats=self.bn_track)), ('elu', nn.ELU()),
            ('maxpool', nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))),
            ('dropout', nn.Dropout(self.dropoutRate))])
        )

        # self.classifier_block = nn.Sequential(
        #     nn.Linear(in_features=200 *
        #                           CalculateOutSize([self.block1, self.block2, self.block3, self.block4],
        #                                            self.Chans, self.Samples),
        #               out_features=self.n_classes,
        #               bias=True))

        self.classifier_block = nn.Sequential(
            nn.Linear(1400,
                      out_features=self.n_classes,
                      bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        output = output.reshape(output.size(0), -1)
        output = self.classifier_block(output)
        # output = F.softmax(output, dim=1)
        return output

    def MaxNormConstraint(self):
        for block in [self.block1, self.block2, self.block3, self.block4]:
            for n, p in block.named_parameters():
                if hasattr(n, 'weight') and (
                        not n.__class__.__name__.startswith('BatchNorm')):
                    p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=2.0)
        for n, p in self.classifier_block.named_parameters():
            if n == '0.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=0.5)





class SquareActivation(nn.Module):
    """y = x²"""
    def forward(self, x):
        return x.pow(2)


class LogActivation(nn.Module):
    """y = log(max(x, eps))  ——  Consistent with ShallowConvNet paper"""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))



class ShallowConvNet(nn.Module):
    def __init__(self,
                 n_classes: int,
                 Chans: int,
                 Samples: int,
                 dropoutRate: Optional[float] = 0.5,
                 bn_track=True,
                 TemporalKernel_Times=1):
        super(ShallowConvNet, self).__init__()
        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        self.bn_track = bn_track  # track_running_state of BatchNorm2d, if fedbs, bn_track = False
        self.TemporalKernel_Times = TemporalKernel_Times  # multiplier for temporal convolution kernel size compared to original

        # self.block1 = nn.Sequential(OrderedDict([
        #     ('conv1', nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(1, 25 * self.TemporalKernel_Times))),
        #     ('conv2', nn.Conv2d(in_channels=40,
        #                         out_channels=40,
        #                         kernel_size=(self.Chans, 1))),
        #     ('bn', nn.BatchNorm2d(num_features=40, track_running_stats=self.bn_track)),
        #     ('activation1', Activation('square')),
        #     ('avgpool', nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))),
        #     ('activation2', Activation('log')),
        #     ('dropout', nn.Dropout(self.dropoutRate))]))

        self.block1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 40, kernel_size=(1, 25 * TemporalKernel_Times))),
            ('conv2', nn.Conv2d(40, 40, kernel_size=(Chans, 1))),
            ('bn',   nn.BatchNorm2d(40, track_running_stats=bn_track)),
            ('act1', SquareActivation()),             
            ('avgp', nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))),
            ('act2', LogActivation()),                   
            ('drop', nn.Dropout(dropoutRate)),
        ]))




        self.classifier_block = nn.Sequential(
            nn.Linear(
                in_features=40 * CalculateOutSize([self.block1], self.Chans, self.Samples),
                out_features=self.n_classes,
                bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = output.reshape(output.size(0), -1)
        output = self.classifier_block(output)
        # output = F.softmax(output, dim=1)
        return output

    def MaxNormConstraint(self):
        for n, p in self.block1.named_parameters():
            if hasattr(n, 'weight') and (
                    not n.__class__.__name__.startswith('BatchNorm')):
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=2.0)
        for n, p in self.classifier_block.named_parameters():
            if n == '0.weight':

                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=0.5)