"""
This file  contains our ml model
"""
import torch
import torch.nn as nn


class LR(nn.Module):

    def __init__(self):
        super(LR, self).__init__()


        self.cnns_2_flatten = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features=7*7*128, out_features=512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=1, bias=True),
        )


    def forward(self, x):
        x = self.cnns_2_flatten(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    data = torch.rand((2, 3, 128, 128))
    lr = LR()
    print(lr(data))
