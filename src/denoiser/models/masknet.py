import torch
import torch.nn as nn
import speechbrain as sb
from speechbrain.lobes.models.transformer.TransformerSE import CNNTransformerSE


class MaskNet(nn.Module):
    def __init__(
        self, 
        in_channels,
        base_channels:int=1024,
        kernel_size:int=3,
        dropout:float=0.1,
        num_layers:int=8,
        d_ffn:int=512,
        nhead:int=12,
        causal:bool=True
    ):
        super(MaskNet, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self.emb_module = sb.nnet.containers.Sequential(
            self.conv_block(in_channels, base_channels),
            self.conv_block(base_channels, base_channels // 2),
            self.conv_block(base_channels // 2, base_channels // 8),
            self.conv_block(base_channels // 8, base_channels // 4)
        )

        self.layer = CNNTransformerSE(
            d_model=in_channels // 2,
            output_size=in_channels,
            output_activation=nn.Sigmoid,
            activation=nn.LeakyReLU,
            dropout=dropout,
            num_layers=num_layers,
            d_ffn=d_ffn,
            nhead=nhead,
            causal=causal,
            custom_emb_module=self.emb_module
        )
    
    def forward(self, x):
        return self.layer(x)
    
    def conv_block(self, in_channels, base_channels):
        return sb.nnet.containers.Sequential(
            sb.nnet.CNN.Conv1d(
                in_channels=in_channels,
                out_channels=base_channels,
                kernel_size=self.kernel_size,
                padding="same"
            ),
            sb.nnet.normalization.LayerNorm(base_channels),
            nn.LeakyReLU(0.01)
        )