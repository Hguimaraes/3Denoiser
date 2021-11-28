import torch.nn as nn

class TEncoderBlock(nn.Module):
    def __init__(self):
        super(TEncoderBlock, self).__init__()
    
    def forward(self, x):
        return x