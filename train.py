from model import UNet
from dataset import SegmentationDataset

import torch.optim as optim
import sys

from torch import manual_seed, cuda
from torch.utils.data import random_split, DataLoader
import torch

import numpy as np


dataset_root = 'data/dataset-sample/'
img_dir = dataset_root + 'image-chips/'
label_dir = dataset_root + 'label-chips/'

epochs = 5
network_width_param = 64
test_set_portion = .2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#OPTIMIZER
lr = .01
momentum = .9
nesterov = True
weight_decay = 5e-4

#LR_SCHEDULER - for MultiStepLR Parameters
milestones = [10,15]
    #list of epoch indeces, must be increasing
gamma = .1

#DATALOADER
batch_size = 16
num_workers = 2

out_channels = 6

def main():
    model = lambda: UNet(in_channels=3, out_channels=out_channels, features=network_width_param)
    #There are six classes, but does that require 5 or 6 channels??
    #width = largest width of any given layer in the network

    optimizer = lambda m: optim.SGD(m.parameters(), lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
    lr_scheduler = lambda o: optim.lr_scheduler.MultiStepLR(o, milestones=milestones, gamma=gamma)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    model = model()
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



def train(model, optimizer, loader, loss_fn, device):
    model.train()

    n_batches = len(loader)
    running_loss = 0.
    with torch.set_grad_enabled(True):
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs, labels = map(lambda x: x.to(device, dtype=torch.float32), (imgs, labels))
            logits = model(imgs)

            #print(logits.dtype, labels.dtype)

            loss = loss_fn(logits, labels)

            """"""""""""""""""""""""

            #print('labels:', type(labels[0]), labels[0].size())
            #print('logits', type(logits), logits[0].size())

            running_loss += loss.item()
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logits.detach()
            print("Batch: {}/{} | Loss: {}".format(batch_idx, n_batches, loss))
            
            '''
            #Detection
            for logit, label in zip(logits, labels):
                performace, _, _ =  score(logits.squeeze(0).numpy(), label=label.squeeze(0).numpy().astype('uint8')), \
                    confusion=True, return_bboxes=False, nms=True)
                counter += np.array(performace, dtype=int)
            progress(counter, running_loss, batch_idx, n_batches)

            '''
    return running_loss / (batch_idx + 1)


def test(model, loader, loss_fn, device):
    model.eval()
    n_batches = len(loader)
    running_loss = 0.
    counter = np.zeroes(shape=5, dtype=int)
    with torch.set_grad_enabled(False):
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs, labels = map(lambda x: x.to(device, dtype=torch.float32), (imgs, labels))
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            running_loss += loss.item()

        '''
        #Detection
        for logit, label in zip(logits, labels):
                performace, _, _ =  score(logits.squeeze(0).numpy(), label=label.squeeze(0).numpy().astype('uint8')), \
                    confusion=True, return_bboxes=False, nms=True)
                counter += np.array(performace, dtype=int)
            progress(counter, running_loss, batch_idx, n_batches)
        '''
        return running_loss / (batch_idx + 1)

    

if(__name__ == "__main__"):
    main()