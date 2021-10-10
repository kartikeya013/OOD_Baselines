import os
import sys
import math
import torch
import pickle
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models
from loguru import logger
from tqdm import tqdm
import torchvision
from torchvision import transforms

from lib.utils.exp import (
    get_model,
    get_modeldir_ens,
    get_transform,
    get_mean, 
    get_std,
    get_dataloader,
)
from lib.model import resnet,lenet
from lib.utils import split_dataloader


parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-i','--ind', type=str, help='in distribution dataset', required=True)
parser.add_argument('-o','--ood', type=str, help='out of distribution dataset', required=True)
parser.add_argument('-m','--model_arch', type=str, help='model architecture', required=True)
parser.add_argument('-b','--batch_size', type=int, default=64)
parser.add_argument('--model_num',type=int, help='the number of classifiers for ensemble',default=5)
parser.add_argument('--dataroot',type=str, help='datatset stroage directory',default='./data/datasets')
# parser.add_argument('--rotation', type=int, default=30, help='rotation angle in case of MNIST dataset')
args = vars(parser.parse_args())
print(args)

modeldir = get_modeldir_ens(args['ind'], args['model_arch'])
ensemble_num = args['model_num']


# ----- load dataset -----
transform = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),])

# transform_iid = get_transform(args['ind'])
# transform_ood = get_transform(args['ood'],rotation=args['rotation'])
std = get_std(args['ind'])
def get_D_iid():
    return torchvision.datasets.ImageFolder("/home/seshank_kartikeya/scratch/imagenet-200/", transform=transform)

def get_D_ood():
    return torchvision.datasets.ImageFolder("/home/seshank_kartikeya/scratch/imagenet-r/", transform=transform)

def split_dataset(dataset):
    n = len(dataset)
    l = torch.utils.data.random_split(dataset, [int(0.05*n),int(0.05*n),n-int(0.05*n)-int(0.05*n)])
    print(len(l[0]),len(l[1]),len(l[2]))
    return torch.utils.data.DataLoader(l[0] , batch_size=args['batch_size'], shuffle=True, num_workers=2),torch.utils.data.DataLoader(l[1] , batch_size=args['batch_size'], shuffle=True, num_workers=2),torch.utils.data.DataLoader(l[2] , batch_size=args['batch_size'], shuffle=True, num_workers=2)
# ind_test_loader = torch.utils.data.DataLoader(get_D(args['ind']) , batch_size=args['batch_size'], shuffle=True, num_workers=2)
# ood_test_loader = torch.utils.data.DataLoader(get_D(args['ood']), batch_size=args['batch_size'], shuffle=False, num_workers=2)
# ind_dataloader_val_for_train, ind_dataloader_val_for_test, ind_dataloader_test = split_dataloader(args['ind'], ind_test_loader, [500,500,-1])
# ood_dataloader_val_for_train, ood_dataloader_val_for_test, ood_dataloader_test = split_dataloader(args['ood'], ood_test_loader, [500,500,-1])
ind_dataloader_val_for_train, ind_dataloader_val_for_test, ind_dataloader_test = split_dataset(get_D_iid())
ood_dataloader_val_for_train, ood_dataloader_val_for_test, ood_dataloader_test = split_dataset(get_D_ood())


# loader = ind_dataloader_test

# mean = 0.
# std = 0.
# for images, _ in loader:
#     batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
#     images = images.view(batch_samples, images.size(1), -1)
#     mean += images.mean(2).sum(0)
#     std += images.std(2).sum(0)

# mean /= len(loader.dataset)
# std /= len(loader.dataset)
print("STD: ",std)

# ----- Calculating and averaging maximum softmax probabilities -----
from lib.inference.ODIN import get_ODIN_score
best_temperature = 1.0
best_magnitude = 0.0

ind_ensemble_val = []
ood_ensemble_val = []
ind_ensemble_test = []
ood_ensemble_test = []
for id, ckpt in enumerate(os.listdir(modeldir)[:ensemble_num]):
    model_path = modeldir + args['ind'] + '_' + args['model_arch'] + f'_{id}.pth'
    model = get_model(args['ind'], args['model_arch'], target_model_path=model_path)

    ind_scores_val_for_train = get_ODIN_score(model, ind_dataloader_val_for_train, best_magnitude, best_temperature, std=std)
    ood_scores_val_for_train = get_ODIN_score(model, ood_dataloader_val_for_train, best_magnitude, best_temperature, std=std)
    ind_ensemble_val.append(ind_scores_val_for_train)
    ood_ensemble_val.append(ood_scores_val_for_train)

    ind_scores_test = get_ODIN_score(model, ind_dataloader_test, best_magnitude, best_temperature, std=std)
    ood_scores_test = get_ODIN_score(model, ood_dataloader_test, best_magnitude, best_temperature, std=std)
    ind_ensemble_test.append(ind_scores_test)
    ood_ensemble_test.append(ood_scores_test)

take_mean_and_reshape = lambda x: np.array(x).mean(axis=0).reshape(-1, 1)
ind_val, ood_val, ind_test, ood_test = map(take_mean_and_reshape, [ind_ensemble_val, ood_ensemble_val, ind_ensemble_test, ood_ensemble_test])

# ----- Train OoD detector using validation data -----
from lib.metric import get_metrics, train_lr
lr = train_lr(ind_val, ood_val)

# ----- Calculating metrics using test data -----
metrics = get_metrics(lr, ind_test, ood_test, acc_type="best")
print("metrics: ", metrics)

