from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from utils import *

from models.resnet import ResNet18, ResNet34, ResNet50
from models.resnext import ResNeXt29, ResNeXt29_1d
from models.resnet import ResNet26_bottle, ResNet38_bottle, ResNet50_bottle
from models.vgg import VGG11

parser = argparse.ArgumentParser(description='Evaluate Bias and Variance on OOD Dataset')
parser.add_argument('--modeldir', type=str, help='folder to save model and training log)')
parser.add_argument('--outdir', type=str, help='folder to save bias and variance results)')
parser.add_argument('--arch', type=str, default='resnet34',
                    help='choose the archtecure from [resnet18/resnet34/resnet50/resnext/resnext_small/vgg]')
parser.add_argument('--trial', default=2, type=int, help='how many trails to run')
parser.add_argument("--device_name", type=str, default="cpu", help="training device")
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/mnist]')
parser.add_argument('--eval-epoch', default=0, type=int, help='choose which epoch to evaluate')
parser.add_argument('--width', default=10, type=int, help='width of resnet')
parser.add_argument('--test-size', default=750000, type=int, help='number of test points')
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
# set up cifar10/cifar10-c test dataset and loader
##################################################
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]

Y_ood_list = []
for index_c in range(len(CORRUPTIONS)):
    if index_c == 0:
        X_ood = np.load('./data/CIFAR-10-C/{}.npy'.format(CORRUPTIONS[index_c]))[:50000]
    else:
        X_ood_idx = np.load('./data/CIFAR-10-C/{}.npy'.format(CORRUPTIONS[index_c]))[:50000]
        X_ood = np.concatenate((X_ood, X_ood_idx), axis=0)
    Y_ood = np.load('./data/CIFAR-10-C/labels.npy')[:50000]
    for i in range(50000):
        Y_ood_list.append(Y_ood[i])

print('X (OOD) shape: ', X_ood.shape)
print('Y (OOD) shape: ', len(Y_ood_list))

testset.data = X_ood
testset.targets = Y_ood_list
testloader = torch.utils.data.DataLoader(testset, batch_size=500, shuffle=False, num_workers=2)

# loss definition
criterion = nn.MSELoss(reduction='mean').cuda()
# variables for bias-variance calculation
NUM_CLASSES = 10
OUTPUST_SUM = torch.Tensor(args.test_size, NUM_CLASSES).zero_().cuda()
OUTPUTS_SUMNORMSQUARED = torch.Tensor(args.test_size).zero_().cuda()
# train/test accuracy/loss
TRAIN_ACC_SUM = 0.0
TEST_ACC_SUM = 0.0
TRAIN_LOSS_SUM = 0.0
TEST_LOSS_SUM = 0.0


def test(net, testloader):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.long().cuda()
            targets_onehot = torch.FloatTensor(targets.size(0), NUM_CLASSES).cuda()
            targets_onehot.zero_()
            targets_onehot.scatter_(1, targets.view(-1, 1).long(), 1)
            outputs = net(inputs)
            loss = criterion(outputs, targets_onehot)
            test_loss += loss.item() * outputs.numel()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    return test_loss / total, 100. * correct / total


def compute_bias_variance(net, testloader, trial):
    net.eval()
    bias2 = 0
    variance = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            targets_onehot = torch.FloatTensor(targets.size(0), NUM_CLASSES).cuda()
            targets_onehot.zero_()
            targets_onehot.scatter_(1, targets.view(-1, 1).long(), 1)
            outputs = net(inputs)
            OUTPUST_SUM[total:(total + targets.size(0)), :] += outputs
            OUTPUTS_SUMNORMSQUARED[total:total + targets.size(0)] += outputs.norm(dim=1) ** 2.0

            bias2 += (OUTPUST_SUM[total:total + targets.size(0), :] / (trial + 1) - targets_onehot).norm() ** 2.0
            variance += OUTPUTS_SUMNORMSQUARED[total:total + targets.size(0)].sum()/(trial + 1) - (OUTPUST_SUM[total:total + targets.size(0), :]/(trial + 1)).norm() ** 2.0
            total += targets.size(0)

    return bias2 / total, variance / total


# Evaluate
for trial in range(args.trial):
    ##########################################
    # set up model
    ##########################################
    if args.arch == 'resnet18':
        net = ResNet18(width=args.width).cuda()
    elif args.arch == 'resnet34':
        net = ResNet34(width=args.width).cuda()
    elif args.arch == 'resnet50':
        net = ResNet50(width=args.width).cuda()
    elif args.arch == 'resnext':
        net = ResNeXt29(width=args.width).cuda()
    elif args.arch == 'resnext_1d':
        net = ResNeXt29_1d(width=args.width).cuda()
    elif args.arch == 'vgg':
        net = VGG11(width=args.width).cuda()
    elif args.arch == 'resnet26_bottle':
        net = ResNet26_bottle(width=args.width).cuda()
    elif args.arch == 'resnet38_bottle':
        net = ResNet38_bottle(width=args.width).cuda()
    elif args.arch == 'resnet50_bottle':
        net = ResNet50_bottle(width=args.width).cuda()
    else:
        print('no available arch')
        raise RuntimeError

    net.load_state_dict(torch.load(os.path.join(args.modeldir, 'model_width{}_trial{}.pkl'.format(args.width, trial))))
    net.eval()

    test_loss, test_acc = test(net, testloader)
    TEST_LOSS_SUM += test_loss
    TEST_ACC_SUM += test_acc

    # compute bias and variance
    bias2, variance = compute_bias_variance(net, testloader, trial)
    variance_unbias = variance * args.trial / (args.trial - 1.0)
    bias2_unbias = TEST_LOSS_SUM / (trial + 1) - variance_unbias
    print('trial: {}, test loss: {:.6f}, test acc: {}, bias2: {}, variance: {}'.format(
        trial, TEST_LOSS_SUM / (trial + 1), TEST_ACC_SUM / (trial + 1), bias2_unbias, variance_unbias))
    log(logfilename, "{}\t{:.5}\t{:.5}\t{:.5}\t{:.5}".format(
        trial, TEST_LOSS_SUM / (trial + 1), TEST_ACC_SUM / (trial + 1), bias2_unbias, variance_unbias))

print('Program finished', flush=True)
