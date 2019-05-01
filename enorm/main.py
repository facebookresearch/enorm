# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import warnings
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn

import models
from data import load_data
from enorm import ENorm


parser = argparse.ArgumentParser(description='ENorm')
parser.add_argument('--dataset', default='cifar10',
                    choices=['cifar10'], help='Specify the dataset')
parser.add_argument('--data-path', default='data/cifar/',
                    help='path to dataset')
parser.add_argument('--n-iter', type=int, default=1,
                    help='Number of runs over which to average results')
parser.add_argument('--model-type', default='linear',
                    choices=['linear', 'conv'])
parser.add_argument('--n-layers', type=int, default=1,
                    help='Number of layers in the model')
parser.add_argument('-j', '--workers', default=2, type=int,
                    help='Number of data loading workers')
parser.add_argument('--epochs', default=60, type=int,
                    help='Number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='Manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='Minibatch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='Initial learning rate')
parser.add_argument('--schedule', default='linear', choices=['linear'],
                    help='Learning rate schedule')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum when using SGD')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='Weight decay')
parser.add_argument('--enorm', default=0, type=float,
                    help='Parameter c. If c = 0, ENorm is not applied')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--save', default='', type=str,
                    help='path to latest checkpoint')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')


def main():
    # get arguments
    global args
    args = parser.parse_args()

    # distributed and GPU
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    # average results over n-iter runs
    top1 = AverageMeter()

    for i in range(args.n_iter):
        print('Iteration {}'.format(i))

        # model definition
        if args.model_type == 'linear':
            model = models.__dict__['FullyConnected'](n_layers=args.n_layers)
        else:
            model = models.__dict__['FullyConvolutional']()

        if args.gpu is not None:
            model = model.cuda(args.gpu)
        else:
            model = torch.nn.DataParallel(model).cuda()

        # loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        # enorm
        enorm = None
        if args.enorm > 0:
            enorm = ENorm(model.named_parameters(), optimizer, args.model_type,
                          args.enorm)

        cudnn.benchmark = True

        # data loading code
        train_loader, _, test_loader = load_data(args.dataset, args.model_type,
            args.batch_size, args.workers, args.data_path)

        # training loop
        for epoch in range(args.start_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch, args.schedule)
            train(train_loader, model, criterion, optimizer, epoch, enorm)
            top1_temp = test(test_loader, model, criterion)

        top1.update(top1_temp)

    print('Average top 1 precision over {} runs: {top1.avg:.2f}'.format(args.n_iter, top1=top1))


def train(train_loader, model, criterion, optimizer, epoch, enorm):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    cpt = 0
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        cpt += input.size(0)
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # enorm
        if enorm is not None:
            enorm.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Speed {3:.3f} ({4:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'Count {5}'.format(
                   epoch, i, len(train_loader),
                   args.batch_size / batch_time.val,
                   args.batch_size / batch_time.avg,
                   cpt,
                   batch_time=batch_time, data_time=data_time,
                   loss=losses, top1=top1, top5=top5))


def test(test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    cpt = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(test_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {2:.3f} ({3:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      'Count {4}'.format(
                       i, len(test_loader),
                       args.batch_size / batch_time.val,
                       args.batch_size / batch_time.avg,
                       cpt,
                       batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        return top1.avg


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """
    def __init__(self, store_values=False):
        self.store_values = store_values
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

        if self.store_values:
            self.values = []

    def update(self, val, n=1):
        if type(val) == list:
            self.values.extend(val)
            self.val = sum(val) / len(val)
            self.sum += sum(val)
            self.count += len(val)
        else:
            self.val = val
            self.sum += val * n
            self.count += n

        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, schedule):
    """
    Decay of the learning rate.
    """
    if schedule == 'linear':
        lr = args.lr * (args.epochs - epoch) / args.epochs
    elif schedule == 'quadratic':
        lr = args.lr * ((args.epochs - epoch) / args.epochs) ** 2
    else:
        lr = args.lr ** (epoch // 30 + 1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
