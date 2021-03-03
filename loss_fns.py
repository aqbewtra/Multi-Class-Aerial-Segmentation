import torch.nn as nn
import torch

import torch.nn.functional as F



#https://github.com/zijundeng/pytorch-semantic-segmentation/
class CrossEntropyLoss2d(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = torch.nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(torch.nn.functional.log_softmax(inputs), targets)


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

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, logits, labels):
        logits = logits.view(-1)
        labels = labels.view(-1)
        return 1 - ((2. * (logits * labels).sum() + 1.) / (logits.sum() + labels.sum() + 1.))