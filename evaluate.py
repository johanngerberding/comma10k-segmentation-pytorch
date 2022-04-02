import numpy as np


def IoU(pred, target):
    "Calculate IoU score"
    intersection = np.logical_and(target, pred)
    union = np.logical_or(target, pred)
    return np.sum(intersection) / np.sum(union)

