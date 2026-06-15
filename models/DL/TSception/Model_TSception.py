import torch
import torch.nn as nn


class TSception(nn.Module):
    def conv_block(self, in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_chan,
                out_channels=out_chan,
                kernel_size=kernel,
                stride=step,
            ),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)),
        )

    def __init__(
        self,
        num_classes,
        input_size,
        sampling_rate,
        num_T,
        num_S,
        hidden,
        dropout_rate,
    ):
        # input_size: 1 x EEG channel x datapoint
        super(TSception, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8
        self.Tception1 = self.conv_block(
            1,
            num_T,
            (1, int(self.inception_window[0] * sampling_rate)),
            1,
            self.pool,
        )
        self.Tception2 = self.conv_block(
            1,
            num_T,
            (1, int(self.inception_window[1] * sampling_rate)),
            1,
            self.pool,
        )
        self.Tception3 = self.conv_block(
            1,
            num_T,
            (1, int(self.inception_window[2] * sampling_rate)),
            1,
            self.pool,
        )

        self.Sception1 = self.conv_block(
            num_T,
            num_S,
            (int(input_size[1]), 1),
            1,
            int(self.pool * 0.25),
        )
        self.Sception2 = self.conv_block(
            num_T,
            num_S,
            (int(input_size[1] * 0.5), 1),
            (int(input_size[1] * 0.5), 1),
            int(self.pool * 0.25),
        )
        self.fusion_layer = self.conv_block(num_S, num_S, (3, 1), 1, 4)
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)
        self.BN_fusion = nn.BatchNorm2d(num_S)

        self.fc = nn.Sequential(
            nn.Linear(num_S, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_ = z
        z = self.Sception2(out)
        out_ = torch.cat((out_, z), dim=2)
        out = self.BN_s(out_)
        out = self.fusion_layer(out)
        out = self.BN_fusion(out)
        out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)
        out = self.fc(out)
        return out
