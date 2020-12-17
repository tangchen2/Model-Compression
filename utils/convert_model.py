import torch
import torch.nn as nn
import sys
sys.path.append("..")
import torch.onnx
from models.preresnet import *
import pickle
import numpy as np

# A model class instance (class not shown)
checkpoint_path = "YOUR_MODEL_PATH"
cfg_path = "YOUR_CFG_MODEL_PATH"  # only need in when u convert pruned model or kd model
with open(cfg_path, 'rb') as f:
    cfg = pickle.load(f)
cfg = np.array(cfg)
net = ResNet(depth=36, dataset='cifar10', cfg=cfg)

net.load_state_dict(torch.load(checkpoint_path)["state_dict"])
# Create the right input shape (e.g. for an image)
input = torch.randn(1, 3, 32, 32)

onnx_path = "MODEL_EXPORT_PATH"
torch.onnx.export(net, input, onnx_path, verbose=True)