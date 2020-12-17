import math
import torch
import torch.nn as nn
from torch.autograd import Variable



defaultcfg = {
    11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, slim_channel="normal", init_weights=True, cfg=None):
        # slim channel used to cut channels directly
        super(VGG, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        assert slim_channel in ["normal", "half", "quarter"]
        self.feature = self.make_layers(cfg, slim_channel, True)

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100

        if slim_channel == "normal":
            final_input = cfg[-1]
        elif slim_channel == "half":
            final_input = cfg[-1] // 2
        else:
            final_input = cfg[-1] // 4

        self.classifier = nn.Linear(final_input, num_classes)
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, slim_channel="normal", batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if slim_channel == 'normal':
                    output_channels = v
                elif slim_channel == 'half':
                    output_channels = v // 2
                else:
                    output_channels = v // 4
                conv2d = nn.Conv2d(in_channels, output_channels, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = output_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        # x = x.view(x.size(0), -1)
        # mnn doesn't support view op? => squeeze instead
        x = torch.squeeze(x)
        x = torch.squeeze(x)

        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    net = VGG(depth=16, slim_channel="quarter")
    for m in net.modules():
        print(m)
    x = Variable(torch.FloatTensor(16, 3, 32, 32))
    y = net(x)
    # print(y.data.shape)
