import os
import sys
import pickle

import numpy as np
import torch
import torchvision
from torchvision import transforms
from loguru import logger

from lib.utils.exp import (
    get_model,
    get_transform,
    get_mean, 
    get_std,
    get_dataloader,
    get_img_size,
    get_inp_channel,
)
from lib.utils import split_dataloader
import argparse


parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-i','--ind', type=str, help='in distribution dataset', required=True)
parser.add_argument('-o','--ood', type=str, help='out of distribution dataset', required=True)
parser.add_argument('-m','--model_arch', type=str, help='model architecture', required=True)
parser.add_argument('--dataroot',type=str, help='datatset stroage directory',default='./data/datasets')
parser.add_argument('--batch_size',type=int,default=512)
parser.add_argument('--inp_process', action='store_true', help='whether do input pre-processing')

args = vars(parser.parse_args())
print(args)

# # ----- load pre-trained model -----
# model = get_model(args['ind'], args['model_arch'])

# # ----- load dataset -----
# transform_iid = get_transform(args['ind'])
# transform_ood = get_transform(args['ood'],rotation=args['rotation'])
# std = get_std(args['ind'])
# img_size = get_img_size(args['ind'])
# inp_channel = get_inp_channel(args['ind'])
# batch_size = args['batch_size'] # recommend: 64 for ImageNet, CelebA, MS1M
# input_process = args['inp_process']

# ind_test_loader = get_dataloader(args['ind'], transform_iid, "test", dataroot=args['dataroot'], batch_size=batch_size)
# ood_test_loader = get_dataloader(args['ood'], transform_ood, "test", dataroot=args['dataroot'], batch_size=batch_size)
# ind_dataloader_val_for_train, ind_dataloader_val_for_test, ind_dataloader_test = split_dataloader(args['ind'], ind_test_loader, [500, 500, -1], random=True)
# ood_dataloader_val_for_train, ood_dataloader_val_for_test, ood_dataloader_test = split_dataloader(args['ood'], ood_test_loader, [500,500, -1], random=True)


# ----- load pre-trained model -----
model = get_model(args['ind'], args['model_arch'])

# ----- load dataset -----
transform = transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor(),])
img_size = 28
inp_channel = 3
batch_size = args['batch_size'] 
input_process = args['inp_process']
# transform_ood = transforms.Compose([transforms.ToTensor(),])
# std = get_std(args['ind']) ##TODO
# ind_test_loader = get_dataloader(args['ind'], transform_iid, "test",dataroot=args['dataroot'],batch_size=args['batch_size'])
# ood_test_loader = get_dataloader(args['ood'], transform_ood, "test",dataroot=args['dataroot'],batch_size=args['batch_size'])
def get_D_iid():
    return torchvision.datasets.ImageFolder("./data/datasets/imagenet-a/", transform=transform)

def get_D_ood():
    return torchvision.datasets.ImageFolder("./lib/training/imagenetr_training/data/imagenet-r/", transform=transform)

def split_dataset(dataset):
    n = len(dataset)
    l = torch.utils.data.random_split(dataset, [int(0.05*n),int(0.05*n),n-int(0.05*n)-int(0.05*n)])
    print(len(l[0]),len(l[1]),len(l[2]))
    return torch.utils.data.DataLoader(l[0] , batch_size=64, shuffle=True, num_workers=2),torch.utils.data.DataLoader(l[1] , batch_size=64, shuffle=True, num_workers=2),torch.utils.data.DataLoader(l[2] , batch_size=64, shuffle=True, num_workers=2)
# ind_test_loader = torch.utils.data.DataLoader(get_D(args['ind']) , batch_size=args['batch_size'], shuffle=True, num_workers=2)
# ood_test_loader = torch.utils.data.DataLoader(get_D(args['ood']), batch_size=args['batch_size'], shuffle=False, num_workers=2)
# ind_dataloader_val_for_train, ind_dataloader_val_for_test, ind_dataloader_test = split_dataloader(args['ind'], ind_test_loader, [500,500,-1])
# ood_dataloader_val_for_train, ood_dataloader_val_for_test, ood_dataloader_test = split_dataloader(args['ood'], ood_test_loader, [500,500,-1])
ind_dataloader_val_for_train, ind_dataloader_val_for_test, ind_dataloader_test = split_dataset(get_D_iid())
ood_dataloader_val_for_train, ood_dataloader_val_for_test, ood_dataloader_test = split_dataset(get_D_ood())


loader = ind_dataloader_test

mean = 0.
std = 0.
for images, _ in loader:
    batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)

mean /= len(loader.dataset)
std /= len(loader.dataset)
print("STD: ",std)


from lib.inference import get_feature_dim_list
from lib.inference.FSS import (
        compute_fss,
        get_FSS_score_ensem,
        get_FSS_score_ensem_process,
        search_FSS_hyperparams
    )
from lib.metric import get_metrics, train_lr

# ----- Calcualte FSS -----
feature_dim_list,_ = get_feature_dim_list(model, img_size, inp_channel, flat=True)
fss = compute_fss(model, len(feature_dim_list), img_size, inp_channel)
layer_indexs = list(range(len(feature_dim_list)))

# ----- Calculate best magnitude for input pre-processing -----
if input_process:
    best_magnitude = search_FSS_hyperparams(model,
                                fss,
                                layer_indexs,
                                ind_dataloader_val_for_train, 
                                ood_dataloader_val_for_train,
                                ind_dataloader_val_for_test, 
                                ood_dataloader_val_for_test, 
                                std=std)

# ----- Calculate FSSD -----
if not input_process: # when no input pre-processing is used
    print('Get FSSD for in-distribution validation data.')
    ind_features_val_for_train = get_FSS_score_ensem(model, ind_dataloader_val_for_train, fss, layer_indexs)
    print('Get FSSD for OoD validation data.')
    ood_features_val_for_train = get_FSS_score_ensem(model, ood_dataloader_val_for_train, fss, layer_indexs)


    print('Get FSSD for in-distribution test data.')
    ind_features_test = get_FSS_score_ensem(model, ind_dataloader_test, fss, layer_indexs)
    print('Get FSSD for OoD test data.')
    ood_features_test = get_FSS_score_ensem(model, ood_dataloader_test, fss, layer_indexs)[:len(ind_features_test)]
else: # when input pre-processing is used
    print('Get FSSD for in-distribution validation data.')
    ind_features_val_for_train = get_FSS_score_ensem_process(model, ind_dataloader_val_for_train, fss, layer_indexs, best_magnitude, std)
    print('Get FSSD for OoD validation data.')
    ood_features_val_for_train = get_FSS_score_ensem_process(model, ood_dataloader_val_for_train, fss, layer_indexs, best_magnitude, std)


    print('Get FSSD for in-distribution test data.')
    ind_features_test = get_FSS_score_ensem_process(model, ind_dataloader_test, fss, layer_indexs, best_magnitude, std)
    print('Get FSSD for OoD test data.')
    ood_features_test = get_FSS_score_ensem_process(model, ood_dataloader_test, fss, layer_indexs, best_magnitude, std)[:len(ind_features_test)]

# ----- Training OoD detector using validation data -----
lr = train_lr(ind_features_val_for_train, ood_features_val_for_train)


# ----- Calculating metrics using test data -----
logger.info("ind_features_test shape: {}".format(ind_features_test.shape))
logger.info("ood_features_test shape: {}".format(ood_features_test.shape))

metrics = get_metrics(lr, ind_features_test, ood_features_test, acc_type="best")
print("metrics:", metrics)
