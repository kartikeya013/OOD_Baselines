'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from resnet import ResNet50,ResNet34
import os, sys
if os.isatty(sys.stdout.fileno()):
    from utils import progress_bar
else:
    progress_bar = print
# import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--weight_decay', '-w', default=5e-4, type=float, help='weight_decay')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()
print(args)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),])

def get_D_iid():
    return torchvision.datasets.ImageFolder("/home/seshank_kartikeya/scratch/imagenet-200/", transform=transform)

def get_D_ood():
    return torchvision.datasets.ImageFolder("/home/seshank_kartikeya/scratch/imagenet-r/", transform=transform)

trainloader = torch.utils.data.DataLoader(get_D_iid() , batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(get_D_ood() , batch_size=128, shuffle=True, num_workers=2)



# for i,batch in enumerate(trainloader):
#     images,label = batch
#     plt.imshow(np.transpose(images[0].numpy(), (1, 2, 0)))
 
# plt.show()

# classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# Model
print('==> Building model..')
net = ResNet34(num_classes=200) 
netName = 'resnet_imageneta'
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}.pth'.format(netName))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

# Training
def train(epoch):
    global best_acc
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    file1 = open("myfile.txt", "w") 
    file1.write(str(epoch)+"\n")
    file1.close()
    acc = 100.*correct/total
    if acc > best_acc:
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

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    

for epoch in range(start_epoch, start_epoch+50):
    train(epoch)
    test(epoch)

optimizer = optim.SGD(net.parameters(), lr=args.lr/10,
                      momentum=0.9, weight_decay=args.weight_decay)

for epoch in range(start_epoch+50, start_epoch+75):
    train(epoch)
    test(epoch)

optimizer = optim.SGD(net.parameters(), lr=args.lr/100,
                      momentum=0.9, weight_decay=args.weight_decay)

for epoch in range(start_epoch+75, start_epoch+100):
    train(epoch)
    test(epoch)
