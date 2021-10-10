# -*- coding: utf-8 -*-
"""Untitled8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Y0Xz8SUJTsto25uBoav-U_gf63upoyR1
"""

# Commented out IPython magic to ensure Python compatibility.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
#import numpy as np
#import matplotlib
from PIL import Image
import numpy as np
import matplotlib
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import sys
import resnet
# %matplotlib inline
import torchvision
import torchvision.transforms as transforms
if os.isatty(sys.stdout.fileno()):
    from utils import progress_bar
else:
    progress_bar = print


transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    # transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
])


def get_D_iid():
    return torchvision.datasets.ImageFolder("/home/seshank_kartikeya/scratch/imagenet-200/", transform=transform)

def get_D_ood():
    return torchvision.datasets.ImageFolder("/home/seshank_kartikeya/scratch/imagenet-r/", transform=transform)


trainloader = torch.utils.data.DataLoader(get_D_iid() , batch_size=64, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(get_D_ood() , batch_size=64, shuffle=True, num_workers=2)




device = ('cuda' if torch.cuda.is_available() else 'cpu')


import resnet
n = 5
nets_ops = []
for i in range(n):
    net = resnet.ResNet34(num_classes=200)
    net = net.to(device)
    if device == 'cuda':
      net = torch.nn.DataParallel(net)
      cudnn.benchmark = True
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    nets_ops.append((net, optimizer))

# Train 
best_acc = 0
def train_model(net, optimizer, netName):
    global best_acc
    eps = .01*8
    alpha = .5
    running_loss = []
    loss_func = nn.CrossEntropyLoss(reduction="mean")
    for epoch in range(15):
        epoch_loss = 0
        print(netName+": Epoch " + str(epoch))
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            x = torch.tensor(inputs, dtype = torch.float, requires_grad=True)
            y = torch.tensor(targets, dtype = torch.float)
            optimizer.zero_grad()
            outputs = net(x)
            loss = loss_func(outputs, targets)
            loss.backward(retain_graph=True)
            train_loss += loss.item()
            x_a = x + eps*(torch.sign(x.grad.data))
            optimizer.zero_grad()

            output_a = net(x_a)

            loss = alpha*loss_func(outputs, targets) + (1-alpha)*loss_func(output_a, targets)
            loss.backward()

            optimizer.step()
            _, predicted = outputs.max(1)
            epoch_loss += loss.item()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # print('Saving..')
        acc = 100.*correct/total
        print("Loss is " + str(epoch_loss) + " " + str(loss)+" Acc: "+str(acc))
        # if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}.pth'.format(netName))
        best_acc = acc
        running_loss.append(epoch_loss/len(inputs))

count = -1
for net, op in nets_ops:
    count+=1
    train_model(net, op, 'imagenet-200_resnet_'+str(count)+'.pth')

