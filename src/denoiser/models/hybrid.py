import torch.nn as nn

from denoiser.models import TEncoderBlock
from denoiser.models import TDecoderBlock
from denoiser.models import CEncoderBlock
from denoiser.models import CDecoderBlock

class HybridDenoiser(nn.Module):
    def __init__(self, depth):
        super(HybridDenoiser, self).__init__()
        TEnc, TDec = None, None
        CEnc, CDec = None, None
    
    def forward(self, x):
        return x