# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn


class FullyConnected(nn.Module):
    """
    Fully connected architetcures for the cifar10 flattened dataset. Stacks
    linear layers interleaved with ReLUs and BatchNorms.

    Args:
        - n_layers: the total number of layers
        - p: dimension of the intermediary layers, i.e. intermediary weight
          matrices have size p x p

    Note:
        - The first layer has size 3072 x p (3072 = 3*32*32), the intermediary
          layers have size p x p and the last layer has size p x 10
    """

    def __init__(self, n_layers=2, p=500):
        super(FullyConnected, self).__init__()
        self.n_layers = n_layers
        self.p = p
        # build architecrture
        for i in range(self.n_layers):
            in_features = 3072 if i == 0 else self.p
            out_features = self.p if i < self.n_layers - 1 else 10
            setattr(self, 'layer_' + '{0:02d}'.format(i), nn.Linear(in_features, out_features, bias=False))
            if i < self.n_layers - 1:
                setattr(self, 'bn_' + '{0:02d}'.format(i), nn.BatchNorm1d(out_features))
            if i < self.n_layers - 1:
                setattr(self, 'relu_' + '{0:02d}'.format(i), nn.ReLU())
        # reset parameters
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        x = input
        for i in range(self.n_layers):
            x = getattr(self, 'layer_' + '{0:02d}'.format(i))(x)
            if i < self.n_layers - 1:
                x = getattr(self, 'bn_' + '{0:02d}'.format(i))(x)
            if i < self.n_layers - 1:
                x = getattr(self, 'relu_' + '{0:02d}'.format(i))(x)
        return x
