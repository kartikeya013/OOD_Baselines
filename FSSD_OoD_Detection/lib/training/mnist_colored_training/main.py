'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import argparse
from PIL import Image
from resnet import ResNet34
import os, sys
if os.isatty(sys.stdout.fileno()):
    from utils import progress_bar
else:
    progress_bar = print
# import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import numpy as np
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
])
class ReaderDataset(Dataset):
    def __init__(self, filename):
        self.model  = torch.load(filename)

    def __len__(self):
        return len(self.model)

    def __getitem__(self, index):
        img, target = self.model[index]        
        img = transform(img)
        
        return img, target

def color_grayscale_arr_train(arr,label):
  """Converts grayscale image to either red or green"""
  assert arr.ndim == 2
  dtype = arr.dtype
  h, w = arr.shape
  arr = np.reshape(arr, [h, w, 1])
  arr = np.array(arr)
  if label<5:
    arr = np.concatenate([arr,(arr*(153/255)).astype(np.uint8),(arr*(51/255)).astype(np.uint8)], axis=2)
  else:
    arr = np.concatenate([(arr*(51/255)).astype(np.uint8),(arr*(51/255)).astype(np.uint8),arr], axis=2)
  return arr

def color_grayscale_arr_test(arr,label):
  """Converts grayscale image to either red or green"""
  assert arr.ndim == 2
  dtype = arr.dtype
  h, w = arr.shape
  arr = np.reshape(arr, [h, w, 1])
  arr = np.array(arr)
  if label<5:
    arr = np.concatenate([(arr*(51/255)).astype(np.uint8),(arr*(51/255)).astype(np.uint8),arr], axis=2)
  else:
    arr = np.concatenate([arr,(arr*(153/255)).astype(np.uint8),(arr*(51/255)).astype(np.uint8)], axis=2)
  return arr

def get_MNISTA():
  root_path    = './data'
  train_mnist = torchvision.datasets.mnist.MNIST(root_path, train=True, download=True)
  train_set = []
  dset_train =  DataLoader(train_mnist, batch_size=128, shuffle=True)
  for idx, (im,label) in enumerate(train_mnist):
    im_array = np.array(im)
    colored_arr = color_grayscale_arr_train(im_array,label)
    train_set.append((Image.fromarray(colored_arr), label)) 

  n = len(train_set)
  
  torch.save(train_set, os.path.join("./data", 'MNIST_A_train.pt'))
  return ReaderDataset("./data/MNIST_A_train.pt")


def get_MNISTB():
  root_path    = './data'
  train_mnist = torchvision.datasets.mnist.MNIST(root_path, train=True, download=True)
  train_set = []
  dset_train =  DataLoader(train_mnist, batch_size=128, shuffle=True)
  for idx, (im,label) in enumerate(train_mnist):
    im_array = np.array(im)
    colored_arr = color_grayscale_arr_test(im_array,label)
    train_set.append((Image.fromarray(colored_arr), label)) 

  n = len(train_set)
  
  torch.save(train_set, os.path.join("./data", 'MNIST_B_test.pt'))
  return ReaderDataset("./data/MNIST_B_test.pt")


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--weight_decay', '-w', default=5e-4, type=float, help='weight_decay')
# parser.add_argument('--rotation',default=30,type=int, help='angle for rotation')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()
print(args)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

transform_test = transforms.Compose([
    # transforms.RandomRotation((args.rotation,args.rotation)),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = get_MNISTA()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = get_MNISTB()
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)


# for i,batch in enumerate(trainloader):
#     images,label = batch
#     plt.imshow(np.transpose(images[0].numpy(), (1, 2, 0)))
 
# plt.show()

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# Model
print('==> Building model..')
net = ResNet34() 
netName = 'resnet_mnist'
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

# Training
def train(epoch):
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

def test(epoch):
    global best_acc
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

    # Save checkpoint.
    file1 = open("myfile.txt", "w")  # append mode
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

for epoch in range(start_epoch, start_epoch+5):
    train(epoch)
    test(epoch)

optimizer = optim.SGD(net.parameters(), lr=args.lr/10,
                      momentum=0.9, weight_decay=args.weight_decay)

for epoch in range(start_epoch+5, start_epoch+10):
    train(epoch)
    test(epoch)

optimizer = optim.SGD(net.parameters(), lr=args.lr/100,
                      momentum=0.9, weight_decay=args.weight_decay)

for epoch in range(start_epoch+10, start_epoch+15):
    train(epoch)
    test(epoch)
