import torch.nn as nn
import torch

import torch.nn.functional as F

from sklearn.metrics import jaccard_similarity_score as jsc

class Jaccard(nn.Module):
    def __init__(self):
        super(Jaccard, self).__init__()

    def forward(self, img, label):
        label = label.cpu().numpy().reshape(-1)
        img = img.cpu().numpy().reshape(-1)
        return jsc(img,label)



def iou(pred, target, n_classes = 12):
  ious = []
  pred = pred.view(-1)
  target = target.view(-1)

  # Ignore IoU for background class ("0")
  for cls in xrange(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
    union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      ious.append(float(intersection) / float(max(union, 1)))
  return np.array(ious)