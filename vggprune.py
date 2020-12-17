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


# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--depth', type=int, default=16,
                    help='depth of the vgg')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')


#####################
# params
#####################
args = parser.parse_args()
test_batch_size = 256
batch_size = 64
pretrain_weight = "YOUR_SR_MODEL_PATH"
prune_model_save_path = "YOUR_PRUNE_MODEL_SAVE_PATH"
data_root = "YOUR_CIFAR_PATH"

cpu_device = "cpu"
gpu_device = "cuda"
#####################
# model
#####################
model = VGG(dataset='cifar10', depth=args.depth)

# model.to(gpu_device)

if pretrain_weight:
    if os.path.isfile(pretrain_weight):
        print("=> loading checkpoint '{}'".format(pretrain_weight))
        checkpoint = torch.load(pretrain_weight)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(pretrain_weight, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.model))

origin_model_acc = best_prec1
ori_model_parameters = sum([param.nelement() for param in model.parameters()])

# origin model calc time
random_input = torch.rand((1, 3, 32, 32)).to(cpu_device)
model.to(cpu_device)
origin_forward_time, origin_flops, origin_params = calc_time_and_flops(random_input, model)
model.to(gpu_device)

#######################
# pre process
#######################
# determine prune mask
total = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]

bn = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn[index:(index+size)] = m.weight.data.abs().clone()
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

pruned_ratio = pruned/total

print('Pre-processing Successful!')


# simple test model after Pre-processing prune (simple set BN scales to zeros)
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


fake_prune_acc = test(model)

########################
# real prune model
########################
newmodel = VGG(dataset='cifar10', cfg=cfg)
newmodel.to(gpu_device)

num_parameters = sum([param.nelement() for param in newmodel.parameters()])

layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.BatchNorm2d):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        layer_id_in_cfg += 1
        start_mask = end_mask.clone()
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        # print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
        w1 = w1[idx1.tolist(), :, :, :].clone()
        m1.weight.data = w1.clone()
    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        m1.weight.data = m0.weight.data[:, idx0].clone()
        m1.bias.data = m0.bias.data.clone()

# conv-bn merge
fuse_model_vgg(newmodel)
prune_model_parameters = sum([param.nelement() for param in newmodel.parameters()])

print("save pruned merged model...")
torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, prune_model_save_path)
#torch.save(newmodel.state_dict(), prune_model_save_path)

real_prune_acc = test(newmodel)


# ##########################
# time and flops
# ##########################
newmodel.to(cpu_device)
pruned_forward_time, pruned_flops, pruned_params = calc_time_and_flops(random_input, newmodel)
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
print("origin accuracy {}   vs    prune accuracy {}".format(best_prec1, real_prune_acc))
print("fake prune acc {}".format(fake_prune_acc))