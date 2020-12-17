import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models.vgg import VGG
from utils.merge_bn import *
import time
from ptflops import get_model_complexity_info


#############################
# utils function
#############################
def calc_time_and_flops(input, model, repeat=50):
    model.eval()
    start = time.time()
    with torch.no_grad():
        for i in range(repeat):
            output = model(input)
        avg_infer_time = (time.time() - start) / repeat
        flops, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True,
                                                  print_per_layer_stat=True)  # 不用写batch_size大小，默认batch_size=1
    return avg_infer_time, flops, params


def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True}

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_root, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
        batch_size=test_batch_size, shuffle=True, **kwargs)
    model.eval()
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    # print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
    #     correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


#####################
# params
#####################
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--depth', type=int, default=16,
                    help='depth of the vgg')
parser.add_argument('--arch', default='vgg', type=str,
                    help='architecture to use')
parser.add_argument('--slim_channel', default="normal", type=str,
                    help='directly cut channels')


args = parser.parse_args()
test_batch_size = 256
batch_size = 64
pretrain_weight = "MODEL_TO_EVALUATE"
cfg_weight = "PRUNE_MODEL_PATH"  # only need in pruned model or kd model, we need cfg info to build net
cfg_checkpoint = torch.load(cfg_weight)

data_root = "YOUR_CIFAR_PATH"

cpu_device = "cpu"
gpu_device = "cuda"
#####################
# model
#####################
if args.arch == "vgg":
    if cfg_weight:
        model = VGG(depth=args.depth, slim_channel=args.slim_channel, cfg=cfg_checkpoint['cfg'])
    else:
        model = VGG(depth=args.depth, slim_channel=args.slim_channel)
else:
    if cfg_weight:
        model = ResNet(depth=args.depth, cfg=cfg_checkpoint['cfg'])
    else:
        model = ResNet(depth=args.depth)

# if you evaluate origin model(no prune)
# comment this code
if args.arch == "resnet":
    fuse_model_resnet(model)
else:
    fuse_model_vgg(model)
# model.to(gpu_device)

if pretrain_weight:
    if os.path.isfile(pretrain_weight):
        print("=> loading checkpoint '{}'".format(pretrain_weight))
        checkpoint = torch.load(pretrain_weight)
#        args.start_epoch = checkpoint['epoch']
#        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'"
              .format(pretrain_weight))
    else:
        print("=> no checkpoint found at '{}'".format(args.model))

ori_model_parameters = sum([param.nelement() for param in model.parameters()])

# origin model calc time
random_input = torch.rand((1, 3, 32, 32)).to(cpu_device)
model.to(cpu_device)
origin_forward_time, origin_flops, origin_params = calc_time_and_flops(random_input, model)
model.to(gpu_device)

acc = test(model)

print("=========== accuracy: {} ==============".format(acc))
print("=========== infer time: {} ==============".format(origin_forward_time))
print("=========== params: {} ==============".format(origin_params))
