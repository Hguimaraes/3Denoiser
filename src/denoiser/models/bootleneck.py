import torch.nn as nn
import speechbrain as sb
from denoiser.models.TEncoder import TEncoderBlock
from denoiser.models.TDecoder import TDecoderBlock
from speechbrain.lobes.models.transformer.Conformer import ConformerEncoderLayer

class Bootleneck(nn.Module):
    def __init__(self, base_ch, depth):
        super(Bootleneck, self).__init__()
        num_ch = (2**(depth-1))*base_ch
        self.lstm = sb.nnet.RNN.LSTM(
            input_size=num_ch,
            hidden_size=num_ch//2,
            num_layers=2,
            bidirectional=True
        )
    
    def forward(self, x):
        x, _ = self.lstm(x)
        return x