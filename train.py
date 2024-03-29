from model import UNet
from dataset import SegmentationDataset

import torch.optim as optim
import sys

from torch import manual_seed, cuda
from torch.utils.data import random_split, DataLoader
import torch
import numpy as np

from glob import glob
import os

from loss_fns import DiceLoss
from nestedUNet import NestedUNet

from jaccard import Jaccard

dataset_root = 'data/dataset-sample/'
img_dir = dataset_root + 'image-chips/'
label_dir = dataset_root + 'label-chips/'

epochs = 2
network_width_param = 64
test_set_portion = .2

gpu_cuda = torch.cuda.is_available()
device = torch.device('cuda' if gpu_cuda else 'cpu')


#OPTIMIZER
lr = .03
momentum = .9
nesterov = True
weight_decay = 5e-4

#LR_SCHEDULER - for MultiStepLR Parameters
milestones = [2,4,6]
gamma = .1

#DATALOADER
batch_size = 16
num_workers = 0
out_channels = 6

save = True

weight = torch.FloatTensor([.2, .05, .2, .05, .4, .1])

def main():
    print("Using CUDA:      {}".format(gpu_cuda))
    model = lambda: NestedUNet(in_channels=3, out_channels=out_channels, filters=network_width_param)

    optimizer = lambda m: optim.SGD(m.parameters(), lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
    # optimizer = lambda m: optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)

    lr_scheduler = lambda o: optim.lr_scheduler.CosineAnnealingWarmRestarts(o, T_0=1, T_mult=1, eta_min=0.0001, last_epoch=-1)
    # lr_scheduler = lambda o: optim.lr_scheduler.MultiStepLR(o, milestones=milestones, gamma=gamma)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    #loss_fn = DiceLoss()

    iou = Jaccard()



    model = model()

    if gpu_cuda:
        model = model.cuda()

    optimizer = optimizer(model)
    lr_scheduler = lr_scheduler(optimizer)

    best_metrics = dict()
    best_metrics['loss'] = sys.maxsize
    for item in ('precision', 'recall', 'f1_score', 'pixel_acc'):
            best_metrics[item] = 0.0

    dataset = SegmentationDataset(img_dir, label_dir, scale=1)

    n_test = int(len(dataset) * test_set_portion)
    n_train = len(dataset) - n_test
    manual_seed(101)
    train_set, test_set = random_split(dataset, [n_train, n_test])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, \
        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, \
        num_workers=num_workers, pin_memory=torch.cuda.is_available())

    
    for epoch in range(epochs):

        print('------- EPOCH [{} / {}] -------'.format(epoch + 1, epochs))

        train_loss = train(model, optimizer, train_loader, loss_fn, device, iou)

        print("Average Loss = {}".format(train_loss))

        test_loss = test(model, test_loader, loss_fn, device, iou)
        print("Test Loss = {}".format(test_loss))
        lr_scheduler.step()

    path = 'saved_models/model-'
    if(save == True):
        path += str(len(glob(os.path.join('saved_models/', '*.pth'))))
        path += '.pth'
        torch.save(model, path)
        print("Train Complete: Model Saved as " + path)



def train(model, optimizer, loader, loss_fn, device, iou):
    model.train()

    n_batches = len(loader)
    running_loss = 0.
    with torch.set_grad_enabled(True):
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs, labels = map(lambda x: x.to(device, dtype=torch.float32), (imgs, labels))
            if gpu_cuda:
                logits = model(imgs).cuda()
            else:
                logits = model(imgs)

            loss = loss_fn.forward(logits.squeeze(0), labels.to(dtype=torch.long)//40)

            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logits.detach()
            
            # print(torch.argmax(logits, dim=1).shape, labels.shape)
            
            print("Batch: {}/{} | Loss: {} | IoU: {} | LR: {}".format(batch_idx + 1, n_batches, loss, iou.forward(torch.argmax(logits, dim=1)[0], labels[0]), get_lr(optimizer)))

    return running_loss / (n_batches)


def test(model, loader, loss_fn, device, iou):
    model.eval()
    n_batches = len(loader)
    running_loss = 0.

    with torch.set_grad_enabled(False):
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs, labels = map(lambda x: x.to(device, dtype=torch.float32), (imgs, labels))
            
            if gpu_cuda:
                logits = model(imgs).cuda()
            else:
                logits = model(imgs)

            try:
                loss = loss_fn.forward(logits.squeeze(0), labels.to(dtype=torch.long))
                running_loss += loss.item()
            except:
                print("EXCEPTION CAUGHT: Test Batch Skipped")

    return running_loss / (n_batches)
    



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if(__name__ == "__main__"):
    main()