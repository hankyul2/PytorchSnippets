import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == '__main__':
    """
    This is criterion exercise in pytorch.

    - `nn.CrossEntropy` is just -log(p). CrossEntropyLoss and BCELoss have difference in how to calculate probabilities. CrossEntropyLoss calculate probability using softmax, but BCELoss needs to use sigmoid yourself and BCEWithLogitsLoss use sigmoid internally.
    - `nn.BCELoss` is -ylog(p) + (y-1)log(1-p). You can calculate multi-label loss in BCELoss.
    - `nn.BCEWithLogitsLoss` is -ylog(sigmoid(p)) + (y-1)log(1-sigmoid(p)) You can calculate multi-label loss in BCEWithLogitsLoss.

    Two related issues
    - LabelSmoothing is cross entropy and binary cross entropy. Same concept is applied to both loss: Target with smoothing probability. 
        - In CrossEntropy, loss is calculated by summing up -ylog(p) + mean(log(p_i)) -> This is multi-class problem
        - In BCEWithLogitsLoss, loss is calculated by summing up every -ylog(sigmoid(x)) + (1-y)log(1-sigmoid(x)) -> This is multi-label problem

    - BEC Target Threshold (timm) is made to remove low probability target used with cutmix or mixup. 

    
    """
    logits = torch.rand([3, 5])
    y = torch.tensor([2, 3, 4])
    y_one_hot = F.one_hot(y, num_classes=5).float()

    ce_loss = nn.CrossEntropyLoss()(logits, y)
    bce_loss = nn.BCELoss()(F.sigmoid(logits), y_one_hot)
    bce_with_logits_loss = nn.BCEWithLogitsLoss()(logits, y_one_hot)

    print(f'cross entropy loss: {ce_loss.detach().item():.4f}')
    print(f'bce loss: {bce_loss.detach().item():.4f}')
    print(f'bce with logits loss: {bce_with_logits_loss.detach().item():.4f}')
