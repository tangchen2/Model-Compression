import torch
import torchvision.transforms as transforms
import torchvision
from torchvision import datasets
import torch.nn.functional as F
import numpy as np
import argparse
from models.preresnet import *
from utils.merge_bn import *


##########################
# parser
##########################
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')

args = parser.parse_args()
##########################
# train params
##########################
device = torch.device("cuda")
student_weight = "STUDENT_MODEL"
teacher_weight = "TEACHER_MODEL"
epochs = 200
best_acc = 0.

batch_size = 64
test_batch_size = 256

temperature = 50
alpha = 0.9

model_save_path = "YOUR_KD_MODEL_SAVE_PATH"
print("================= temperature: {}, alpha: {}, lr: {} prune: 0.5========================".format(temperature, alpha, args.lr))
##########################
# data set
##########################
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

##########################
# build model
##########################
student_checkpoint = torch.load(student_weight)
student = ResNet(depth=38, dataset='cifar10', cfg=student_checkpoint["cfg"])
fuse_model_resnet(student)
student.load_state_dict(student_checkpoint["state_dict"])
student.to(device)

teacher_checkpoint = torch.load(teacher_weight)
teacher = ResNet(depth=92, dataset='cifar10', cfg=None)
teacher.load_state_dict(teacher_checkpoint["state_dict"])
teacher.to(device)

hard_criterion = torch.nn.CrossEntropyLoss()
soft_criterion = torch.nn.KLDivLoss()

optimizer = torch.optim.SGD(student.parameters(),
                            lr=args.lr, momentum=0.9, weight_decay=5e-4)


##########################
# define train & test
##########################
def train(epoch):
    student.train()
    teacher.eval()

    train_loss = 0
    total = 0
    correct = 0
    for batch_data, batch_label in train_loader:
        optimizer.zero_grad()

        batch_data = batch_data.cuda()
        batch_label = batch_label.cuda()

        student_output = student(batch_data)
        teacher_output = teacher(batch_data)

        hard_loss = hard_criterion(student_output, batch_label)
        soft_loss = soft_criterion(F.log_softmax(student_output/temperature, dim=1),
                            F.softmax(teacher_output/temperature, dim=1)) * temperature * temperature
        loss = hard_loss * (1.-alpha) + soft_loss * alpha
#        print("hard loss: {}, soft loss: {}, total loss: {}".format(hard_loss, soft_loss, loss))
        _, student_predict = student_output.max(1)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total += batch_data.size(0)
        correct += student_predict.eq(batch_label).sum().item()

    train_loss = train_loss / (len(train_loader) + 1)
    train_acc = correct / total * 100.
    print("Epoch {}, Train Loss {}, Correct / Total: {} / {}, Train Accuracy {}%".format(epoch, train_loss, correct, total, train_acc))


def test(epoch):
    global best_acc
    student.eval()
    teacher.eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for batch_data, batch_label in test_loader:
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)

            student_output = student(batch_data)
            _, student_predict = student_output.max(1)

            correct += student_predict.eq(batch_label).sum().item()
            total += batch_data.size(0)

    test_acc = correct / total * 100.
    print("Epoch {}, Correct / Total: {} / {}, Test Accuracy {}%".format(epoch, correct, total, test_acc))

    if test_acc > best_acc:
        print("saving best test acc...")
        state = {
            "state_dict": student.state_dict(),
            "acc": test_acc
        }
        torch.save(state, model_save_path)
        best_acc = test_acc


for i in range(epochs):
    # 更新学习率
    if i in [int(epochs * 0.5), int(epochs*0.75)]:
        for param_group in optimizer.param_groups:
            param_group["lr"] *= 0.1

    train(i)
    test(i)
