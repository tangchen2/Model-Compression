import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *
from utils.merge_bn import *
import pickle
import time
from ptflops import get_model_complexity_info


############################
# eval function
############################
def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_root, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
        batch_size=test_batch_size, shuffle=False, **kwargs)

    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    # print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
    #     correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


def calc_time_and_flops(input, model, repeat=20):
    model.eval()
    start = time.time()
    with torch.no_grad():
        for i in range(repeat):
            output = model(input)
        avg_infer_time = (time.time() - start) / repeat
        flops, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True,
                                                  print_per_layer_stat=True)  # 不用写 batch_size大小，默认batch_size=1

    return avg_infer_time, flops, params


############################
# parser
############################
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--depth', type=int, default=164,
                    help='depth of the resnet')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')

args = parser.parse_args()

model = ResNet(depth=args.depth, dataset='cifar10')

#######################
# params
#######################
model_path = "YOUR_SR_MODEL_PATH"
data_root = "YOUR_CIFAR_PATH"
model_save_path = "YOUR_MODEL_SAVE_PATH"
cpu_device = torch.device("cpu")
device = torch.device("cuda")
test_batch_size = 256
#######################
# load checkpoint
#######################
if os.path.isfile(model_path):
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    args.start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
          .format(model_path, checkpoint['epoch'], best_prec1))
else:
    print("model path wrong...")

ori_model_acc = best_prec1
ori_model_parameters = sum([param.nelement() for param in model.parameters()])

random_input = torch.rand((1, 3, 32, 32)).to(cpu_device)
model.to(cpu_device)
origin_forward_time, origin_flops, origin_params = calc_time_and_flops(random_input, model)

############################
# prune mask
############################
model.to(device)
total = 0

for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]

bn = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn[index:(index + size)] = m.weight.data.abs().clone()
        index += size

y, i = torch.sort(bn)
thre_index = int(total * args.percent)
thre = y[thre_index]

pruned = 0
cfg = []
cfg_mask = []
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        weight_copy = m.weight.data.abs().clone()
        mask = weight_copy.gt(thre).float().cuda()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
              format(k, mask.shape[0], int(torch.sum(mask))))
    elif isinstance(m, nn.MaxPool2d):
        cfg.append('M')

pruned_ratio = pruned / total
# simple test model after Pre-processing prune (simple set BN scales to zeros)
fake_prune_acc = test(model)

############################
# prune mask
############################
new_model = ResNet(depth=args.depth, dataset='cifar10', cfg=cfg)
new_model.to(device)

prune_model_parameters = sum([param.nelement() for param in new_model.parameters()])

old_modules = list(model.modules())
new_modules = list(new_model.modules())
layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
conv_count = 0

for layer_id in range(len(old_modules)):
    m0 = old_modules[layer_id]
    m1 = new_modules[layer_id]
    if isinstance(m0, nn.BatchNorm2d):

        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))

        if isinstance(old_modules[layer_id + 1], channel_selection):
            # If the next layer is the channel selection layer, then the current batchnorm 2d layer won't be pruned.
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()

            # We need to set the channel selection layer.
            m2 = new_modules[layer_id + 1]
            m2.indexes.data.zero_()
            m2.indexes.data[idx1.tolist()] = 1.0

            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]
        else:
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):
        if conv_count == 0:
            m1.weight.data = m0.weight.data.clone()
            conv_count += 1
            continue
        if isinstance(old_modules[layer_id - 1], channel_selection) or isinstance(old_modules[layer_id - 1],
                                                                                  nn.BatchNorm2d):
            # This convers the convolutions in the residual block.
            # The convolutions are either after the channel selection layer or after the batch normalization layer.
            conv_count += 1
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()

            # If the current convolution is not the last convolution in the residual block, then we can change the
            # number of output channels. Currently we use `conv_count` to detect whether it is such convolution.
            if conv_count % 3 != 1:
                w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            continue

        # We need to consider the case where there are downsampling convolutions.
        # For these convolutions, we just copy the weights.
        m1.weight.data = m0.weight.data.clone()
    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))

        m1.weight.data = m0.weight.data[:, idx0].clone()
        m1.bias.data = m0.bias.data.clone()

real_prune_model_acc = test(new_model)

fuse_model_resnet(new_model)
real_prune_fuse_model_acc = test(new_model)

#####################
# save model
#####################
torch.save({'cfg': cfg, 'state_dict': new_model.state_dict()}, model_save_path)


# ##########################
# time and flops
# ##########################
new_model.to(cpu_device)
pruned_forward_time, pruned_flops, pruned_params = calc_time_and_flops(random_input, new_model)
print("origin net forward time {}   vs   prune net forward time {}".format(
    origin_forward_time, pruned_forward_time
))
print("origin net GFLOPS {}   vs prune net GFLOPS {}".format(
    origin_flops, pruned_flops
))
print("origin net params {}   vs   prune net params {}".format(
    origin_params, pruned_params
))

print("origin net params {}  vs   prune net params {}".format(ori_model_parameters, prune_model_parameters))
print("origin accuracy {}   vs    prune accuracy {}".format(ori_model_acc, real_prune_model_acc))

# print("prune config:")
# print(cfg)


