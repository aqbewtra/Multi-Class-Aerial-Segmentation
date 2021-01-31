import torch
import torchvision

import numpy as np
from skimage.morphology import label
from skimage.feature import peak_local_max
from collections import Counter, defaultdict

from .image_transforms import pool

def find_bboxes(logit, padding=0, mass_requirement=0, scores4nms=False):
    local_maximums = peak_local_max(logit)
    dense = np.zeros_like(logit, dtype=bool)
    dense[local_maximums[:, 0], local_maximums[:, 1]] = True
    labels = pool(
        label(dense, background=False).astype('uint8'), 
        kernel_size=2).squeeze(0).numpy()
    obj_ids = np.unique(labels)[1:]
    bboxes = []
    for obj in obj_ids:
        pxls = np.where(labels == obj)
        #print(len(pxls[0]), len(pxls[1]))
        if pxls[0].shape[0] > mass_requirement:
            xmin = np.min(pxls[1]) - padding
            xmax = np.max(pxls[1]) + padding
            ymin = np.min(pxls[0]) - padding
            ymax = np.max(pxls[0]) + padding
            bboxes.append([xmin, ymin, xmax, ymax])
    if not scores4nms:
        return np.array(bboxes), None
    scores = []
    for bbox in bboxes:
        scores.append(np.sum(logit[bbox[1]: bbox[3], bbox[0]: bbox[2]]))
    return torch.Tensor(bboxes), torch.Tensor(scores)

def bbox_confusion(pred, expd):
    confusion = defaultdict(set)
    pred = set([tuple(p) for p in pred])
    expd = set([tuple(e) for e in expd])
    removable = set()
    while len(pred):
        bbox_p = pred.pop()
        found_intersection = False
        for bbox_e in expd:
            x0 = max(bbox_p[0], bbox_e[0])
            y0 = max(bbox_p[1], bbox_e[1])
            x1 = min(bbox_p[2], bbox_e[2])
            y1 = min(bbox_p[3], bbox_e[3])
            intersection = max(0, x1 - x0) * max(0, y1 - y0)
            if intersection > 0:
                union = (x0, y0, x1, y1)
                confusion['true_positives'].add(union)
                removable.add(bbox_e)
                found_intersection = True
        if not found_intersection:
            confusion['false_positives'].add(bbox_p)
        else:
            for bbox in removable:
                expd.remove(bbox)
            removable.clear()
    confusion['false_negatives'].update(expd)
    return confusion

def bbox_confusion_metric(pred, expd):
    confusion = Counter()
    pred = set([tuple(p) for p in pred])
    expd = set([tuple(e) for e in expd])
    removable = set()
    while len(pred):
        bbox_p = pred.pop()
        found_intersection = False
        for bbox_e in expd:
            x0 = max(bbox_p[0], bbox_e[0])
            y0 = max(bbox_p[1], bbox_e[1])
            x1 = min(bbox_p[2], bbox_e[2])
            y1 = min(bbox_p[3], bbox_e[3])
            intersection = max(0, x1 - x0) * max(0, y1 - y0)
            if intersection > 0:
                confusion['true_positives'] += 1
                removable.add(bbox_e)
                found_intersection = True
        if not found_intersection:
            confusion['false_positives'] += 1
        else:
            for bbox in removable:
                expd.remove(bbox)
            removable.clear()
    confusion['false_negatives'] += len(expd)
    return confusion['true_positives'], confusion['false_positives'], confusion['false_negatives']

def score(logit, **kwargs):
    find_kw = lambda key, els: els if key not in kwargs.keys() else kwargs[key]  
    assert isinstance(logit, np.ndarray) and len(logit.shape) == 2
    if len(np.unique(logit)) != 2:
        logit = (logit > 0.8).astype('uint8')
    label = find_kw('label', None)
    return_bboxes, confusion = find_kw('return_bboxes', True), find_kw('confusion', False)
    nms, iou_threshold = find_kw('nms', True), find_kw('iou_threshold', 0.01)
    padding, mass_requirement = find_kw('padding', 0), find_kw('mass_requirement', 0)
    predicted_bboxes, logit_scores = find_bboxes(
        logit, padding=padding, 
        mass_requirement=mass_requirement, scores4nms=nms)
    if nms:
        predicted_bboxes = predicted_bboxes[torchvision.ops.nms(
            boxes=predicted_bboxes, scores=logit_scores, 
            iou_threshold=iou_threshold)].numpy().astype(int)
    if label is not None:
        assert isinstance(label, np.ndarray) and len(label.shape) == 2
        if len(np.unique(label)) != 2:
            label = (label > 200).astype('uint8')
        expected_bboxes, label_scores = find_bboxes(
            label, padding=padding,
            mass_requirement=mass_requirement, scores4nms=nms)
        if nms:
            expected_bboxes = expected_bboxes[torchvision.ops.nms(
                boxes=expected_bboxes, scores=label_scores, 
                iou_threshold=iou_threshold)].numpy().astype(int)
    if confusion:
        if label is None:
            raise ValueError('label must be defined as a nd array in order to \
                create a confusion matrix')
        label_sum = np.sum(label)
        if label_sum == 0:
            and_sum = 0
        else:
            and_sum = np.sum(np.logical_and(logit, label))
        true_positives, false_positives, false_negatives = bbox_confusion_metric(pred=predicted_bboxes, expd=expected_bboxes)
        conf_matrix = np.array([true_positives, false_positives, false_negatives, and_sum, label_sum])
    return conf_matrix if confusion else None, predicted_bboxes if return_bboxes else None,\
         expected_bboxes if (label is not None) and return_bboxes else None
