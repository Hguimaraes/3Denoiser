import torch.nn as nn

class CEncoderBlock(nn.Module):
    def __init__(self):
        super(CEncoderBlock, self).__init__()
    
    def forward(self, x):
        return x