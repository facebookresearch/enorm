# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

class ENorm:
    """
    Implements Equi-normalization for feedforward fully-connected and
    convolutional networks.

    Args:
        - named_params: the named parameters of your model, obtained as
          model.named_parameters()
        - optimizer: the optimizer, necessary for the momentum buffer update
          Note: only torch.optim.SGD supported for the moment
        - model_type: choose among ['linear', 'conv'] (see main.py)
        - c: asymmetric scaling factor that introduces a depth-wise penalty on
          the weights (default:1)
        - p: compute row and columns p-norms (default:2)

    Notes:
        - For all the supported architectures [fully connected, fully
          convolutional], we do not balance the last layer
        - In practice, we have found the training to be more stable when we do
          not balance the biases
    """

    def __init__(self, named_params, optimizer, model_type, c=1, p=2):
        self.named_params = list(named_params)
        self.optimizer = optimizer
        self.model_type = model_type
        self.momentum = self.optimizer.param_groups[0]['momentum']
        self.alpha = 0.5
        self.p = p

        # names to filter out
        to_remove = ['bn']
        fliter_map = lambda x: not any(name in x[0] for name in to_remove)

        # weights and biases
        self.weights = [(n, p) for n, p in self.named_params if 'weight' in n]
        self.weights = list(filter(fliter_map, self.weights))
        self.biases = []
        self.n_layers = len(self.weights)

        # scaling vector
        self.n_layers = len(self.weights)
        self.C = [c] * self.n_layers

    def _get_weight(self, i, orientation='l'):
        _, param = self.weights[i]
        if self.model_type != 'linear':
            if orientation == 'l':
                # (C_in x k x k) x C_out
                param = param.view(param.size(0), -1).t()
            else:
                # C_in x (k x k x C_out)
                param = param.permute(1, 2, 3, 0).contiguous().view(param.size(1), -1)
        return param

    def step(self):
        if self.model_type == 'linear':
            self._step_fc()
        else:
            self._step_conv()

    def _step_fc(self):
        left_norms = self._get_weight(0).norm(p=self.p, dim=1).data
        right_norms = self._get_weight(1).norm(p=self.p, dim=0).data

        for i in range(1, self.n_layers - 1):
            balancer = (right_norms / (left_norms * self.C[i - 1])).pow(self.alpha)

            left_norms = self._get_weight(i).norm(p=self.p, dim=1).data
            right_norms = self._get_weight(i + 1).norm(p=self.p, dim=0).data

            if len(self.biases) > 0: self.biases[i - 1][1].data.mul_(balancer)
            self._get_weight(i - 1).data.t_().mul_(balancer).t_()
            self._get_weight(i).data.mul_(1 / balancer)

    def _step_conv(self):
        left_w = self._get_weight(0, 'l')
        right_w = self._get_weight(1, 'r')

        left_norms = left_w.norm(p=2, dim=0).data
        right_norms = right_w.norm(p=2, dim=1).data

        for i in range(1, self.n_layers - 1):
            balancer = (right_norms / (left_norms * self.C[i-1])).pow(self.alpha)

            left_w = self._get_weight(i, 'l')
            right_w = self._get_weight(i + 1, 'r')

            left_norms = left_w.norm(p=2, dim=0).data
            right_norms = right_w.norm(p=2, dim=1).data

            self.weights[i - 1][1].data.mul_(
                balancer.unsqueeze(1).unsqueeze(2).unsqueeze(3))
            self.weights[i][1].data.mul_(
                1 / balancer.unsqueeze(1).unsqueeze(2).unsqueeze(0))

            if self.momentum:
                self.optimizer.state[self.weights[i - 1][1]]['momentum_buffer'].mul_(
                    1 / balancer.unsqueeze(1).unsqueeze(2).unsqueeze(3))
                self.optimizer.state[self.weights[i][1]]['momentum_buffer'].mul_(
                    balancer.unsqueeze(1).unsqueeze(2).unsqueeze(0))
