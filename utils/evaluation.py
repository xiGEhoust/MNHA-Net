import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve
import numpy as np
import utils.datasets
from torch.utils.data import DataLoader


def cal_confusion_matrix(predict, target, threshold=0.5):
    """compute local confusion matrix for a batch of predict and target masks
    Args:
        predict (_type_): _description_
        target (_type_): _description_
        region (_type_): _description_
        
    Returns:
        TP, TN, FP, FN
    """
    predict = (predict > threshold).float()
    TP = torch.sum(predict * target, dim=(1, 2, 3))
    TN = torch.sum((1-predict) * (1-target), dim=(1, 2, 3))
    FP = torch.sum(predict * (1-target), dim=(1, 2, 3))
    FN = torch.sum((1-predict) * target, dim=(1, 2, 3))
    return TP, TN, FP, FN

def cal_F1(TP, TN, FP, FN):
    """compute F1 score for a batch of TP, TN, FP, FN
    Args:
        TP (_type_): _description_
        TN (_type_): _description_
        FP (_type_): _description_
        FN (_type_): _description_
    """
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    F1 = 2 * precision * recall / (precision + recall + 1e-8)
    # F1 = torch.mean(F1) # fuse the Batch dimension
    return F1
    
