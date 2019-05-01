# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def load_data(dataset, model_type, batch_size, nb_workers, data_path):
    """
    Loads data from CIFAR10.

    Args:
        - dataset: cifar10
        - model_type: choose among ['fc', 'conv'] (see main.py)
        - batch_size: train and test batch size
        - nb_workers: number of workers for dataloader
        - data_path: path to dataset. As the dataset is automatically
          downloaded, this can be set to 'data/cifar/'

    """

    if dataset == 'cifar10':
        # transforms
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        if model_type == 'linear':
            flatten = transforms.Lambda(lambda x: x.view(-1))
            transform_train = transforms.Compose([transforms.ToTensor(), normalize, flatten])
            transform_val_test = transform_train

        elif model_type == 'conv':
            transform_train = transforms.Compose([
                transforms.RandomCrop(28, padding=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            transform_val_test = transforms.Compose([
                transforms.CenterCrop(28),
                transforms.ToTensor(),
                normalize
            ])

        else:
            raise NotImplementedError(dataset, model_type)

        # train set
        train_set = datasets.CIFAR10(
               root=data_path, train=True, download=True, transform=transform_train)
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(40960))
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, sampler=train_sampler, num_workers=nb_workers)

        # val set
        val_set = datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=transform_val_test)
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(40960, 50000))
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=batch_size, sampler=val_sampler, num_workers=nb_workers)

        # test
        test_set = datasets.CIFAR10(
            root=data_path, train=False, download=True, transform=transform_val_test)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=nb_workers)

        return train_loader, val_loader, test_loader

    else:
        raise NotImplementedError(dataset, model_type)
