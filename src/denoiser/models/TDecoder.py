import torch.nn as nn

class TDecoderBlock(nn.Module):
    def __init__(self):
        super(TDecoderBlock, self).__init__()
    
    def forward(self, x):
        return x