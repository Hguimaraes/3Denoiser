import torch.nn as nn
import speechbrain as sb
from denoiser.models.TEncoder import TEncoderBlock
from denoiser.models.TDecoder import TDecoderBlock
from speechbrain.lobes.models.transformer.Conformer import ConformerEncoder

class Bootleneck(nn.Module):
    def __init__(self, base_ch, depth):
        super(Bootleneck, self).__init__()
        num_ch = (2**(depth-1))*base_ch
        self.layer = ConformerEncoder(
            d_ffn=num_ch,
            num_layers=2,
            nhead=8, 
            d_model=num_ch,
            kernel_size=3,
            attention_type='regularMHA'
        )
    
    def forward(self, x):
        x, self_attn = self.layer(x)
        return x