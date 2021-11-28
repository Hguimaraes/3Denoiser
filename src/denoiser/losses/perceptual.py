import torch
import torch.nn as nn

# Perceptual Loss
class PerceptualLoss(nn.Module):
    def __init__(self, PRETRAINED_MODEL_PATH:str, alpha:float=10):
        super(PerceptualLoss, self).__init__()

    def forward(self, y_hat, y, lens=None):
        return 0