# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn


class FullyConvolutional(nn.Module):
    """
    Fully convolutional architecture for cifar10 datset as described in
    Gitman et al., 'Comparison of Batch Normalization and Weight Normalization
    Algorithms for the Large-scale Image Classification'.
    """

    def __init__(self, bias=False):
        super(FullyConvolutional, self).__init__()
        # first block
        self.conv1 = nn.Conv2d(3, 128, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, 1, bias=bias)
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        # second block
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(256, 256, 3, 1, 1, bias=bias)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1, bias=bias)
        self.pool2 = nn.MaxPool2d(3, 2, 1)
        # third block
        self.conv7 = nn.Conv2d(256, 320, 3, 1, 1, bias=bias)
        self.conv8 = nn.Conv2d(320, 320, 1, 1, 0, bias=bias)
        self.conv9 = nn.Conv2d(320, 10, 1, 1, 1, bias=bias)
        self.pool3 = nn.AvgPool2d(5)
        # relu
        self.relu = nn.ReLU()
        # batch norm
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        # reset parameters
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.bn1(x)
        x = self.pool1(self.relu(x))

        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.conv6(x)
        x = self.bn2(x)
        x = self.pool2(self.relu(x))

        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.conv9(x)
        x = self.pool3(self.relu(x))

        return x.view(x.size(0), x.size(1))
