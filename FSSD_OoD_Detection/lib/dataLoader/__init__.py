# from lib.dataLoader.cifar_svhn import *

import os
import sys
import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from PIL import Image
def getTargetDataSet(data_type, batch_size, input_TF, dataroot, split='train',rotation=None):
    if data_type == "cifar10":
        data_loader = getCIFAR10(
            batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1, train=split=='train', val=split=='test'
        )
        return data_loader
    elif data_type == "svhn":
        data_loader = getSVHN(
            batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1, train=split=='train', val=split=='test'
        )
    elif data_type == "fmnist":
        data_loader = getFMNIST(
            batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1, train=split=='train', val=split=='test'
        )
    elif data_type == "mnist":
        data_loader = getMNIST(
            batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1, train=split=='train', val=split=='test'
        )
    elif data_type == "mnist_a":
        data_loader = getMNISTA(
            batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1, train=split=='train', val=split=='test'
        )
    elif data_type == "mnist_b":
        data_loader = getMNISTB(
            batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1, train=split=='train', val=split=='test'
        )
    else:
        raise NotImplementedError

    return data_loader


def getSVHN(
    batch_size,
    TF,
    data_root="/tmp/public_dataset/pytorch",
    train=True,
    val=True,
    **kwargs
):
    data_root = os.path.expanduser(os.path.join(data_root, "svhn-data"))
    num_workers = kwargs.setdefault("num_workers", 1)
    kwargs.pop("input_size", None)

    def target_transform(target):
        new_target = target - 1
        if new_target == -1:
            new_target = 9
        return new_target

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(root=data_root, split="train", download=True, transform=TF),
            batch_size=batch_size,
            shuffle=True,
            **kwargs
        )
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(root=data_root, split="test", download=True, transform=TF),
            batch_size=batch_size,
            shuffle=False,
            **kwargs
        )
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getCIFAR10(
    batch_size,
    TF,
    data_root="/tmp/public_dataset/pytorch",
    train=True,
    val=True,
    **kwargs
):
    data_root = os.path.expanduser(os.path.join(data_root, "cifar10-data"))
    num_workers = kwargs.setdefault("num_workers", 1)
    kwargs.pop("input_size", None)
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=data_root, train=True, download=True, transform=TF),
            batch_size=batch_size,
            shuffle=True,
            **kwargs
        )
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=data_root, train=False, download=True, transform=TF),
            batch_size=batch_size,
            shuffle=False,
            **kwargs
        )
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getFMNIST(
    batch_size,
    TF,
    data_root="/tmp/public_dataset/pytorch",
    train=True,
    val=True,
    **kwargs
):
    data_root = os.path.expanduser(os.path.join(data_root, "fmnist-data"))
    num_workers = kwargs.setdefault("num_workers", 1)
    kwargs.pop("input_size", None)
    ds = []
    if train:
        print("Get FMNIST training data")
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                root=data_root, train=True, download=True, transform=TF
            ),
            batch_size=batch_size,
            shuffle=True,
            **kwargs
        )
        ds.append(train_loader)
    if val:
        print("Get FMNIST validation data")
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                root=data_root, train=False, download=True, transform=TF
            ),
            batch_size=batch_size,
            shuffle=False,
            **kwargs
        )
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    
    return ds

def getMNIST(
    batch_size,
    TF,
    data_root="/tmp/public_dataset/pytorch",
    train=True,
    val=True,
    **kwargs
):
    # data_root = os.path.expanduser(os.path.join(data_root, 'mnist-data'))
    num_workers = kwargs.setdefault("num_workers", 1)
    kwargs.pop("input_size", None)
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=data_root, train=True, download=True, transform=TF),
            batch_size=batch_size,
            shuffle=True,
            **kwargs
        )
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=data_root, train=False, download=True, transform=TF),
            batch_size=batch_size,
            shuffle=False,
            **kwargs
        )
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
])
class ReaderDataset(Dataset):
    def __init__(self, filename):
        self.model  = torch.load(filename)
        data = []
        targetlist = []
        for index in range(len(self.model)):
            img, target = self.model[index]     
            img = transform(img)
            # print(img.unsqueeze(0).size())   
            data.append(img.unsqueeze(0))
            targetlist.append(target)
        self.data = torch.cat([x for x in data])
        self.targets = targetlist
        # print(self.target)

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

def get_MNISTA(path,val):
  root_path    = path
  train_mnist = torchvision.datasets.mnist.MNIST(root_path, train=val, download=True)
  train_set = []
  dset_train =  DataLoader(train_mnist, batch_size=128, shuffle=True)
  for idx, (im,label) in enumerate(train_mnist):
    im_array = np.array(im)
    colored_arr = color_grayscale_arr_train(im_array,label)
    train_set.append((Image.fromarray(colored_arr), label)) 

  n = len(train_set)
  
  torch.save(train_set, os.path.join(path, 'MNIST_A_train.pt'))
  return ReaderDataset(path+"/MNIST_A_train.pt")


def get_MNISTB(path,val):
  root_path    = path
  train_mnist = torchvision.datasets.mnist.MNIST(root_path, train=val, download=True)
  train_set = []
  dset_train =  DataLoader(train_mnist, batch_size=128, shuffle=True)
  for idx, (im,label) in enumerate(train_mnist):
    im_array = np.array(im)
    colored_arr = color_grayscale_arr_test(im_array,label)
    train_set.append((Image.fromarray(colored_arr), label)) 

  n = len(train_set)
  
  torch.save(train_set, os.path.join(path, 'MNIST_B_test.pt'))
  return ReaderDataset(path+"/MNIST_B_test.pt")


def getMNISTA(
    batch_size,
    TF,
    data_root="/tmp/public_dataset/pytorch",
    train=True,
    val=True,
    **kwargs
):
    # data_root = os.path.expanduser(os.path.join(data_root, 'mnist-data'))
    num_workers = kwargs.setdefault("num_workers", 1)
    kwargs.pop("input_size", None)
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            get_MNISTA(path=data_root,val=True),
            batch_size=batch_size,
            shuffle=True,
            **kwargs
        )
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            get_MNISTA(path=data_root,val=False),
            batch_size=batch_size,
            shuffle=False,
            **kwargs
        )
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def getMNISTB(
    batch_size,
    TF,
    data_root="/tmp/public_dataset/pytorch",
    train=True,
    val=True,
    **kwargs
):
    # data_root = os.path.expanduser(os.path.join(data_root, 'mnist-data'))
    num_workers = kwargs.setdefault("num_workers", 1)
    kwargs.pop("input_size", None)
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            get_MNISTB(path=data_root,val=True),
            batch_size=batch_size,
            shuffle=True,
            **kwargs
        )
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            get_MNISTB(path=data_root,val=False),
            batch_size=batch_size,
            shuffle=False,
            **kwargs
        )
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

