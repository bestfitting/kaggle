import sys
sys.path.insert(0, '../..')
import torch.nn as nn
from layers.loss_funcs import lovasz_losses as L

def dice_score(prob, truth, threshold=0.5):
    num = prob.size(0)

    prob = prob > threshold
    truth = truth > 0.5

    prob = prob.view(num, -1)
    truth = truth.view(num, -1)
    intersection = (prob * truth)

    score = 2. * (intersection.sum(1) + 1.).float() / (prob.sum(1) + truth.sum(1) + 2.).float()
    score[score >= 1] = 1
    score = score.sum() / num
    return score

class SymmetricLovaszLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SymmetricLovaszLoss, self).__init__()
    def forward(self, logits, targets, epoch):
        return ((L.lovasz_hinge(logits, targets, per_image=True)) \
                + (L.lovasz_hinge(-logits, 1-targets, per_image=True))) / 2
