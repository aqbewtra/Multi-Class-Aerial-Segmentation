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

dataset_root = 'data/dataset-sample/'
img_dir = dataset_root + 'image-chips/'
label_dir = dataset_root + 'label-chips/'

epochs = 8
network_width_param = 64
test_set_portion = .2

gpu_cuda = torch.cuda.is_available()
device = torch.device('cuda' if gpu_cuda else 'cpu')


#OPTIMIZER
lr = .01
momentum = .9
nesterov = True
weight_decay = 5e-4

#LR_SCHEDULER - for MultiStepLR Parameters
milestones = [2,4]
    #list of epoch indeces, must be increasing
gamma = .1

#DATALOADER
batch_size = 16
num_workers = 0

out_channels = 6

save = True

def main():
    print("Using CUDA:      {}".format(gpu_cuda))
    model = lambda: UNet(in_channels=3, out_channels=out_channels, features=network_width_param)
    #There are six classes, but does that require 5 or 6 channels??
    #width = largest width of any given layer in the network

    #optimizer = lambda m: optim.SGD(m.parameters(), lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
    optimizer = lambda m: optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
    #lr_scheduler = lambda o: optim.lr_scheduler.CosineAnnealingLR(o, 8, eta_min=0, last_epoch=-1)
    lr_scheduler = lambda o: optim.lr_scheduler.CosineAnnealingWarmRestarts(o, T_0=10, T_mult=2, eta_min=0.01, last_epoch=-1)
    #lr_scheduler = lambda o: optim.lr_scheduler.MultiStepLR(o, milestones=milestones, gamma=gamma)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    #loss_fn = WeightedFocalLoss()



    model = model()

    if gpu_cuda:
        model = model.cuda()

    optimizer = optimizer(model)
    lr_scheduler = lr_scheduler(optimizer)

    best_metrics = dict()
    best_metrics['loss'] = sys.maxsize
    for item in ('precision', 'recall', 'f1_score', 'pixel_acc'):
            best_metrics[item] = 0.0

    dataset = SegmentationDataset(img_dir, label_dir, scale=1, mode='nearest')

    n_test = int(len(dataset) * test_set_portion)
    n_train = len(dataset) - n_test
    manual_seed(101)
    train_set, test_set = random_split(dataset, [n_train, n_test])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, \
        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, \
        num_workers=num_workers, pin_memory=torch.cuda.is_available())

    # Cross Entropy Loss for Segmentation
    # Focal Loss
    # Dice Loss
    # This is a way to do pixel level classification
    #Look into nested UNet ----> a
    
    #TRAIN
    
    for epoch in range(epochs):

        print('------- EPOCH [{} / {}] -------'.format(epoch + 1, epochs))

        #train, store metrics
        train_loss = train(model, optimizer, train_loader, loss_fn, device)

        print("Average Loss = {}".format(train_loss))
        #test, store metrics
        test_loss = test(model, test_loader, loss_fn, device)
        print("Test Loss = {}".format(test_loss))
        lr_scheduler.step()

        #update best metrics

        # if best metrics improved, or it is the first epoch, save model

        # display best metrics

    #Save Model    
    path = 'saved_models/model-'
    if(save == True):
        path += str(len(glob(os.path.join('saved_models/', '*.pth'))))
        path += '.pth'
        torch.save(model, path)
        print("Train Complete: Model Saved as " + path)



def train(model, optimizer, loader, loss_fn, device):
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

            #print(logits.size())

            #print(logits.dtype, labels.dtype)

            #loss = loss_fn(logits, labels)
            loss = loss_fn.forward(logits, labels)
            
            """"""""""""""""""""""""

            #print('labels:', type(labels[0]), labels[0].size())
            #print('logits', type(logits), logits[0].size())

            running_loss += loss.item()
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logits.detach()

            #correct += (logits == labels).float().sum()/(logits.shape[0] * logits.shape[1] * logits.shape[2] * logits.shape[3]) * 100

            print("Batch: {}/{} | Loss: {} | LR: {}".format(batch_idx + 1, n_batches, loss, get_lr(optimizer)))
            #print("Accuracy:    {} %".format(int(correct)))

            '''
            #Detection
            for logit, label in zip(logits, labels):
                performace, _, _ =  score(logits.squeeze(0).numpy(), label=label.squeeze(0).numpy().astype('uint8')), \
                    confusion=True, return_bboxes=False, nms=True)
                counter += np.array(performace, dtype=int)
            progress(counter, running_loss, batch_idx, n_batches)

            '''
    return running_loss / (n_batches)


def test(model, loader, loss_fn, device):
    model.eval()
    n_batches = len(loader)
    running_loss = 0.
    #counter = np.zeroes(shape=5, dtype=int)
    with torch.set_grad_enabled(False):
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs, labels = map(lambda x: x.to(device, dtype=torch.float32), (imgs, labels))
            if gpu_cuda:
                logits = model(imgs).cuda()
            else:
                logits = model(imgs)
            #loss = loss_fn(logits, labels)
            loss = loss_fn.forward(logits, labels)
            running_loss += loss.item()

        '''
        #Detection
        for logit, label in zip(logits, labels):
                performace, _, _ =  score(logits.squeeze(0).numpy(), label=label.squeeze(0).numpy().astype('uint8')), \
                    confusion=True, return_bboxes=False, nms=True)
                counter += np.array(performace, dtype=int)
            progress(counter, running_loss, batch_idx, n_batches)
        '''
    return running_loss / (n_batches)
    
'''
#https://amaarora.github.io/2020/06/29/FocalLoss.html

class WeightedFocalLoss(torch.nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        if gpu_cuda:
            self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        else:
            self.alpha = torch.tensor([alpha, 1-alpha])
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
'''

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if(__name__ == "__main__"):
    main()