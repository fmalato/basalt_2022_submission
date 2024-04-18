import torch
import torch.nn as nn

from collections import OrderedDict


class CaveClassifier(nn.Module):

    def __init__(self, in_channels=3, input_width=320, input_height=320):
        super().__init__()
        self.in_channels = in_channels
        self.input_width = input_width
        self.input_height = input_height
        self.conv_block1 = self.conv_block(in_channels1=in_channels, out_channels1=64, kernel_size=3, stride=2,
                                           padding=1)
        self.conv_block2 = self.conv_block(in_channels1=64, out_channels1=64, kernel_size=3, stride=2, padding=1)
        self.conv_block3 = self.conv_block(in_channels1=64, out_channels1=64, kernel_size=3, stride=2, padding=1)
        self.conv_block4 = self.conv_block(in_channels1=64, out_channels1=64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        flat_feats = self.get_flatten_size()
        self.fc1 = nn.Linear(in_features=flat_feats, out_features=1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=1024, out_features=2)
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.conv_block1(x)
        y = self.conv_block2(y)
        y = self.conv_block3(y)
        y = self.conv_block4(y)
        y = self.flatten(y)
        y = self.relu1(self.fc1(y))
        y = self.relu2(self.fc2(y))

        return self.softmax(y)

    def get_flatten_size(self):
        with torch.no_grad():
            x = torch.rand(size=(1, self.in_channels, self.input_width, self.input_height))
            y = self.conv_block1(x)
            y = self.conv_block2(y)
            y = self.conv_block3(y)
            y = self.conv_block4(y)

            _, c, w, h = y.size()

            return c * w * h

    def conv_block(self, in_channels1, out_channels1, kernel_size, stride, padding):
        conv = nn.Sequential(OrderedDict([
            ('conv',
             nn.Conv2d(in_channels=in_channels1, out_channels=out_channels1, kernel_size=kernel_size, stride=stride,
                       padding=padding)),
            ('activation', nn.ReLU())
        ]))

        return conv