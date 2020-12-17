from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models.preresnet import *
from models.vgg import VGG

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--arch', default='vgg', type=str,
                    help='architecture to use')
parser.add_argument('--slim_channel', type=str, default="normal",
                    help="direct cut channels")
parser.add_argument('--depth', default=16, type=int,
                    help='depth of the neural network')

########################
# params
########################
args = parser.parse_args()
momentum = 0.9
weight_decay = 1e-4
epochs = 160
batch_size=64
test_batch_size=256
device = "cuda" if torch.cuda.is_available() else "cpu"
if not args.sr:
    model_save_path = "YOUR_MODEL_SAVE_PATH"
else:
    model_save_path = "YOUR_MODEL_SAVE_PATH"
print("============= arch: {}, depth: {}, slim_channel: {} ================".format(args.arch, args.depth, args.slim_channel))
print("============= sr flag: {}, s ratio: {}, lr: {} =================".format(args.sr, args.s, args.lr))

########################
# data set
########################
cifar_path = "YOUR_CIFAR_PATH"
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(cifar_path, train=True, download=True,
                     transform=transforms.Compose([
                         transforms.Pad(4),
                         transforms.RandomCrop(32),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                     ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(cifar_path, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])),
    batch_size=test_batch_size, shuffle=True, **kwargs)


########################
# network
########################
if args.arch == "vgg":
    model = VGG(depth=args.depth, slim_channel=args.slim_channel)
else:
    # ResNet doesn't support slim channel strategy
    # model = ResNet(depth=args.depth)
    model = ResNet(depth=args.depth)

if torch.cuda.is_available():
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=momentum, weight_decay=weight_decay)

# load pre weight
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))


# additional penalty term of sparsity-induced on bn weights => 0
def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s * torch.sign(m.weight.data))  # L1


def train(epoch):
    model.train()
    train_loss = 0
    total = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        pred = output.max(1, keepdim=True)[1]
        loss.backward()
        if args.sr:
            updateBN()
        optimizer.step()

        total += target.size(0)
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
#        correct += pred.eq(target).sum().item()
        train_loss += loss.item()
    print("train epoch {}, loss {}, acc {}%, correct/total: {}/{}".format(
        epoch, train_loss / (len(train_loader) + 1), 100. * correct / total, correct, total))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
            _, pred = output.max(1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
#            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


best_prec1 = 0.
for epoch in range(epochs):
    if epoch in [epochs * 0.4, epochs * 0.7]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    train(epoch)

    prec1 = test()
    if prec1 > best_prec1:
        best_prec1 = prec1
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, model_save_path)
        print("Best accuracy: " + str(best_prec1))

