from __future__ import print_function
import os
import argparse
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from utils import *
from models.nn_mnist import DNN

parser = argparse.ArgumentParser(description='MNIST Training')
parser.add_argument('--outdir', type=str, default='',
                    help='folder name to specify for saving checkpoint and log)')
parser.add_argument('--trial', default=5, type=int, help='how many trails to run')
parser.add_argument('--dataset', default='fmnist', type=str, help='dataset = [fmnist]')
parser.add_argument('--weight-decay', default=1e-5, type=float, help='weight_decay')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr-decay', type=int, default=100, help='How often to decrease learning by gamma.')
parser.add_argument('--num-epoch', default=200, type=int, help='number of epoches')
parser.add_argument('--width', default=100, type=int, help='width')
parser.add_argument('--test-size', default=10000, type=int, help='number of test points')
parser.add_argument('--save-freq', default=100, type=int, metavar='N', help='save frequency')
parser.add_argument('--gpuid', default='0', type=str)
args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

##################################################
# setup log file
##################################################
outdir = '{}_trial{}_mse{}'.format(args.dataset, args.trial, args.outdir)
if not os.path.exists(outdir):
    os.makedirs(outdir)
logfilename = os.path.join(outdir, 'log_width{}.txt'.format(args.width))
init_logfile(logfilename, "trial\ttrain_loss\ttrain acc\ttest loss\ttest acc\tbias2\tvariance")

##################################################
# set up cifar10 train/test dataset and loader
##################################################
transform_train = transforms.Compose([
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


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


# Training
def train(net, trainloader):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        targets_onehot = torch.FloatTensor(targets.size(0), NUM_CLASSES).cuda()
        targets_onehot.zero_()
        targets_onehot.scatter_(1, targets.view(-1, 1).long(), 1)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets_onehot)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * outputs.numel()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return train_loss / total, 100. * correct / total


# Test
def test(net, testloader):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
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


# set up index after random permutation
permute_index = np.split(np.random.permutation(len(trainset)), args.trial)

for trial in range(args.trial):
    ##########################################
    # set up subsampled cifar10 train loader
    ##########################################
    trainsubset = get_subsample_dataset(trainset, permute_index[trial])
    trainloader = torch.utils.data.DataLoader(trainsubset, batch_size=128, shuffle=True)

    ##########################################
    # set up model and optimizer
    ##########################################
    net = DNN(width=args.width).cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_decay, gamma=0.1)

    for epoch in range(1, args.num_epoch + 1):
        train_loss, train_acc = train(net, trainloader)
        test_loss, test_acc = test(net, testloader)
        print('epoch: {}, train_loss: {:.6f}, train acc: {}, test loss: {:.6f}, test acc: {}'.format(epoch, train_loss, train_acc, test_loss, test_acc))
        scheduler.step(epoch)
        if epoch % args.save_freq == 0:
            torch.save(net.state_dict(), os.path.join(outdir, 'model_width{}_trial{}_epoch{}.pkl'.format(args.width, trial, epoch)))

    TRAIN_LOSS_SUM += train_loss
    TEST_LOSS_SUM += test_loss
    TRAIN_ACC_SUM += train_acc
    TEST_ACC_SUM += test_acc

    # compute bias and variance
    bias2, variance = compute_bias_variance(net, testloader, trial)
    variance_unbias = variance * args.trial / (args.trial - 1.0)
    bias2_unbias = TEST_LOSS_SUM / (trial + 1) - variance_unbias
    print('trial: {}, train_loss: {:.6f}, train acc: {}, test loss: {:.6f}, test acc: {}, bias2: {}, variance: {}'.format(
        trial, TRAIN_LOSS_SUM / (trial + 1), TRAIN_ACC_SUM / (trial + 1), TEST_LOSS_SUM / (trial + 1),
        TEST_ACC_SUM / (trial + 1), bias2_unbias, variance_unbias))
    log(logfilename, "{}\t{:.5}\t{:.5}\t{:.5}\t{:.5}\t{:.5}\t{:.5}".format(
        trial, TRAIN_LOSS_SUM / (trial + 1), TRAIN_ACC_SUM / (trial + 1), TEST_LOSS_SUM / (trial + 1),
        TEST_ACC_SUM / (trial + 1), bias2_unbias, variance_unbias))

    # save the model
    torch.save(net.state_dict(), os.path.join(outdir, 'model_width{}_trial{}.pkl'.format(args.width, trial)))

print('Program finished', flush=True)
