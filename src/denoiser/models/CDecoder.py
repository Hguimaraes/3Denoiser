import torch.nn as nn

class CDecoderBlock(nn.Module):
    def __init__(self):
        super(CDecoderBlock, self).__init__()
    
    def forward(self, x):
        return x