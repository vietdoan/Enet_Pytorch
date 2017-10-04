import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy2d(input, target, weight=None, ignore_index=-100):
    input = F.log_softmax(input)
    criterion = nn.NLLLoss2d(weight=weight, ignore_index=ignore_index)
    loss = criterion(input, target)
    return loss