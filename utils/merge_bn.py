import torch
import torch.nn as nn
import torchvision as tv
from models.preresnet import *
from models.vgg import VGG
import pickle


class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, x):
        # print("Dummy, Dummy.")
        return x


def fuse(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)
    beta = bn.weight
    gamma = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma
    fused_conv = nn.Conv2d(conv.in_channels,
                           conv.out_channels,
                           conv.kernel_size,
                           conv.stride,
                           conv.padding,
                           bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv


def fuse_model_vgg(model):
    for name, child in model.named_children():
        # print("name: {}, child {}".format(name, child))
        if isinstance(child, nn.Sequential):
            for i in range(len(child)):
                if isinstance(child[i], nn.Conv2d) and isinstance(child[i+1], nn.BatchNorm2d):
                    conv_to_merge = child[i]
                    bn_to_merge = child[i+1]
                    fused_conv = fuse(conv_to_merge, bn_to_merge)
                    child[i] = fused_conv
                    child[i+1] = DummyModule()


def fuse_model_resnet(model):
    for name, child in model.named_children():
        # print("name: {}, child {}".format(name, child))
        if isinstance(child, nn.Sequential):
            for block in child:
                # 注意bottleneck内部用.直接访问，不要用index
                conv_to_merge = block.conv1
                bn_to_merge = block.bn2
                fused_conv = fuse(conv_to_merge, bn_to_merge)
                block.conv1 = fused_conv
                block.bn2 = DummyModule()

                conv_to_merge = block.conv2
                bn_to_merge = block.bn3
                fused_conv = fuse(conv_to_merge, bn_to_merge)
                block.conv2 = fused_conv
                block.bn3 = DummyModule()


if __name__ == '__main__':
    # net = ResNet(depth=38)
    # print("origin net {}".format(net))
    # for name, m in net.named_children():
    #     print("name {}, m {}".format(name, m))
    # net = fuse_model(net)
    net = VGG(depth=16)
    print("origin model {}".format(net))
    print("================")
    fuse_model_vgg(net)
    print("fuse model {}".format(net))
    # module_list = list(net.modules())
    # for idx in range(len(module_list)):
    #     print(module_list[idx])