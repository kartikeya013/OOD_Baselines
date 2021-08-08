import os
import sys
import pickle

import numpy as np
import torch
from torchvision import transforms
from loguru import logger

from lib.utils.exp import (
    get_model,
    get_transform,
    get_mean, 
    get_std,
    get_dataloader,
)
from lib.utils import split_dataloader
import argparse
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-i','--ind', type=str, help='in distribution dataset', required=True)
parser.add_argument('-o','--ood', type=str, help='out of distribution dataset', required=True)
parser.add_argument('-m','--model_arch', type=str, help='model architecture', required=True)
parser.add_argument('-b','--batch_size', type=int, help='batch size', default=512)
parser.add_argument('--dataroot',type=str, help='datatset stroage directory', default='./data/datasets')
parser.add_argument('--rotation', type=int, default=30, help='rotation angle in case of MNIST dataset')
args = vars(parser.parse_args())
print(args)

# # ----- load pre-trained model -----
# model = get_model(args['ind'], args['model_arch'])

# # ----- load dataset -----
# transform_iid = get_transform(args['ind'])
# transform_ood = get_transform(args['ood'],rotation=args['rotation'])
# std = get_std(args['ind'])
# ind_test_loader = get_dataloader(args['ind'], transform_iid, "test",dataroot=args['dataroot'],batch_size=args['batch_size'])
# ood_test_loader = get_dataloader(args['ood'], transform_ood, "test",dataroot=args['dataroot'],batch_size=args['batch_size'])
# ind_dataloader_val_for_train, ind_dataloader_val_for_test, ind_dataloader_test = split_dataloader(args['ind'], ind_test_loader, [500,500,-1], random=True)
# ood_dataloader_val_for_train, ood_dataloader_val_for_test, ood_dataloader_test = split_dataloader(args['ood'], ood_test_loader, [500,500,-1], random=True)

# ----- load pre-trained model -----
model = get_model(args['ind'], args['model_arch'])

# ----- load dataset -----
transform = transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor(),])
# transform_ood = transforms.Compose([transforms.ToTensor(),])
# std = get_std(args['ind']) ##TODO
# ind_test_loader = get_dataloader(args['ind'], transform_iid, "test",dataroot=args['dataroot'],batch_size=args['batch_size'])
# ood_test_loader = get_dataloader(args['ood'], transform_ood, "test",dataroot=args['dataroot'],batch_size=args['batch_size'])
def get_D(dataName):
    return torchvision.datasets.ImageFolder("./lib/training/pacs_training/data/kfold/{}".format(dataName), transform=transform)

def split_dataset(dataset):
    n = len(dataset)
    l = torch.utils.data.random_split(dataset, [int(0.05*n),int(0.05*n),n-int(0.05*n)-int(0.05*n)])
    print(len(l[0]),len(l[1]),len(l[2]))
    return torch.utils.data.DataLoader(l[0] , batch_size=args['batch_size'], shuffle=True, num_workers=2),torch.utils.data.DataLoader(l[1] , batch_size=args['batch_size'], shuffle=True, num_workers=2),torch.utils.data.DataLoader(l[2] , batch_size=args['batch_size'], shuffle=True, num_workers=2)
# ind_test_loader = torch.utils.data.DataLoader(get_D(args['ind']) , batch_size=args['batch_size'], shuffle=True, num_workers=2)
# ood_test_loader = torch.utils.data.DataLoader(get_D(args['ood']), batch_size=args['batch_size'], shuffle=False, num_workers=2)
# ind_dataloader_val_for_train, ind_dataloader_val_for_test, ind_dataloader_test = split_dataloader(args['ind'], ind_test_loader, [500,500,-1])
# ood_dataloader_val_for_train, ood_dataloader_val_for_test, ood_dataloader_test = split_dataloader(args['ood'], ood_test_loader, [500,500,-1])
ind_dataloader_val_for_train, ind_dataloader_val_for_test, ind_dataloader_test = split_dataset(get_D(args['ind']))
ood_dataloader_val_for_train, ood_dataloader_val_for_test, ood_dataloader_test = split_dataset(get_D(args['ood']))

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

# ----- Calculate best temperature and magnitude for input pre-processing -----
from lib.inference.ODIN import search_ODIN_hyperparams, get_ODIN_score
logger.info("search ODIN params")
best_temperature, best_magnitude = search_ODIN_hyperparams(model, ind_dataloader_val_for_train, ood_dataloader_val_for_train, ind_dataloader_val_for_test, ood_dataloader_val_for_test, std=std)
print("best params: ", best_temperature, best_magnitude)

# ----- Calculate ODIN score for validation data -----
ind_scores_val_for_train = get_ODIN_score(model, ind_dataloader_val_for_train, best_magnitude, best_temperature, std=std)
ood_scores_val_for_train = get_ODIN_score(model, ood_dataloader_val_for_train, best_magnitude, best_temperature, std=std)
ind_features_val_for_train = ind_scores_val_for_train.reshape(-1,1)
ood_features_val_for_train = ood_scores_val_for_train.reshape(-1,1)

# ----- Calculate ODIN score for test data -----
ind_scores_test = get_ODIN_score(model, ind_dataloader_test, best_magnitude, best_temperature, std=std)
ood_scores_test = get_ODIN_score(model, ood_dataloader_test, best_magnitude, best_temperature, std=std)[:len(ind_scores_test)]
ind_features_test = ind_scores_test.reshape(-1,1)
ood_features_test = ood_scores_test.reshape(-1,1)

# ----- Train OoD detector using validation data -----
from lib.metric import get_metrics, train_lr
lr = train_lr(ind_features_val_for_train, ood_features_val_for_train)

# ----- Calculating metrics using test data -----
metrics = get_metrics(lr, ind_features_test, ood_features_test, acc_type="best")
print("metrics: ", metrics)
