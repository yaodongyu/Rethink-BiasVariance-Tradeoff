from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import os
import argparse
from utils import *

from models.resnet_ce import ResNet18, ResNet34, ResNet50

parser = argparse.ArgumentParser(description='Evaluate Bias and Variance on OOD Dataset')
parser.add_argument('--model-dir-list', nargs='+', type=str)
parser.add_argument('--outdir', type=str, help='folder to save bias and variance results)')
parser.add_argument('--arch', type=str, default='resnet34',
                    help='choose the archtecure from [resnet18/resnet34/resnet50/resnext/resnext_small/vgg]')
parser.add_argument('--trial', default=5, type=int, help='how many trails to run')
parser.add_argument("--device_name", type=str, default="cpu", help="training device")
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/mnist]')
parser.add_argument('--eval-epoch', default=0, type=int, help='choose which epoch to evaluate')
parser.add_argument('--width', default=10, type=int, help='width of resnet')
parser.add_argument('--test-size', default=10000, type=int, help='number of test points')
parser.add_argument('--wr', action='store_true', help='sample with replacement')
parser.add_argument('--gpuid', default='0', type=str)
args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

##################################################
# setup log file
##################################################
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
logfilename = os.path.join(args.outdir, 'log_width{}.txt'.format(args.width))
init_logfile(logfilename, "trial\ttest loss\ttest acc\tbias2\tvariance")

##################################################
# set up cifar10 test dataset and loader
##################################################
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=500, shuffle=False, num_workers=2)

# loss definition
criterion = nn.CrossEntropyLoss().cuda()
nll_loss = nn.NLLLoss(size_average=False)
# variables for bias-variance calculation
NUM_CLASSES = 10
OUTPUTS_LOG_AVG = torch.Tensor(args.test_size, NUM_CLASSES).zero_().cuda()
# train/test accuracy/loss
TRAIN_ACC_SUM = 0.0
TEST_ACC_SUM = 0.0
TRAIN_LOSS_SUM = 0.0
TEST_LOSS_SUM = 0.0


def kl_div_cal(P, Q):
    return (P * (P / Q).log()).sum()


def test(net, testloader):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    return test_loss / total, 100. * correct / total


def compute_log_output_kl(net, testloader):
    net.eval()
    total = 0
    outputs_log_total = torch.Tensor(args.test_size, NUM_CLASSES).zero_().cuda()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            outputs = F.softmax(outputs, dim=1)
            outputs_log_total[total:(total + targets.size(0)), :] = outputs.log()
            total += targets.size(0)
    return outputs_log_total


def compute_normalization_kl(outputs_log_avg):
    outputs_norm = torch.Tensor(args.test_size, NUM_CLASSES).zero_().cuda()
    for idx in range(args.test_size):
        for idx_y in range(NUM_CLASSES):
            outputs_norm[idx, idx_y] = torch.exp(outputs_log_avg[idx, idx_y])
    for idx in range(args.test_size):
        y_total = 0.0
        for idx_y in range(NUM_CLASSES):
            y_total += outputs_norm[idx, idx_y]
        outputs_norm[idx, :] /= (y_total * 1.0)
    return outputs_norm


def compute_bias_variance_kl(net, testloader, outputs_avg):
    net.eval()
    bias2 = 0
    variance = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            outputs = F.softmax(outputs, dim=1)
            bias2 += nll_loss(outputs_avg[total:total + targets.size(0), :].log(), targets)
            for idx in range(len(inputs)):
                variance_idx = kl_div_cal(outputs_avg[total + idx], outputs[idx])
                variance += variance_idx
                assert variance_idx > -0.0001
            total += targets.size(0)
    return bias2 / total, variance / total


##################################################
# compute log-output average
##################################################
for trial in range(args.trial):
    for model_dir in args.model_dir_list:
        ##########################################
        # set up model
        ##########################################
        if args.arch == 'resnet18':
            net = ResNet18(width=args.width).cuda()
        elif args.arch == 'resnet34':
            net = ResNet34(width=args.width).cuda()
        elif args.arch == 'resnet50':
            net = ResNet50(width=args.width).cuda()
        else:
            print('no available arch')
            raise RuntimeError

        net.load_state_dict(torch.load(os.path.join(model_dir, 'model_width{}_trial{}.pkl'.format(args.width, trial))))
        net.eval()
        OUTPUTS_LOG_AVG += compute_log_output_kl(net, testloader) * (1.0 / (args.trial * len(args.model_dir_list)))

##################################################
# normalization
##################################################
OUTPUTS_NORM = compute_normalization_kl(OUTPUTS_LOG_AVG)
variance_total = 0.0

##################################################
# compute bias variance
##################################################
for trial in range(args.trial):
    for model_dir in args.model_dir_list:
        ##########################################
        # set up model
        ##########################################
        if args.arch == 'resnet18':
            net = ResNet18(width=args.width).cuda()
        elif args.arch == 'resnet34':
            net = ResNet34(width=args.width).cuda()
        elif args.arch == 'resnet50':
            net = ResNet50(width=args.width).cuda()
        else:
            print('no available arch')
            raise RuntimeError

        net.load_state_dict(torch.load(os.path.join(model_dir, 'model_width{}_trial{}.pkl'.format(args.width, trial))))
        net.eval()

        test_loss, test_acc = test(net, testloader)
        TEST_LOSS_SUM += test_loss
        TEST_ACC_SUM += test_acc

        # compute bias and variance
        bias2, variance = compute_bias_variance_kl(net, testloader, OUTPUTS_NORM)
        variance_total += variance
        variance_avg = variance_total / (args.trial * len(args.model_dir_list) * 1.0)
        print('trial: {}, test loss: {:.6f}, test acc: {}, bias2: {}, variance: {}'.format(
            trial, TEST_LOSS_SUM / (args.trial * len(args.model_dir_list) * 1.0),
            TEST_ACC_SUM / (args.trial * len(args.model_dir_list) * 1.0), bias2, variance_avg))
        log(logfilename, "{}\t{:.5}\t{:.5}\t{:.5}\t{:.5}".format(
            trial, TEST_LOSS_SUM / (args.trial * len(args.model_dir_list) * 1.0),
            TEST_ACC_SUM / (args.trial * len(args.model_dir_list) * 1.0), bias2, variance_avg))

print('Program finished', flush=True)
