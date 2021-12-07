import torch
import torch.nn as nn
import torch.nn.functional as F
import speechbrain as sb

class Upsampling(nn.Module):
    def __init__(self, factor):
        super(Upsampling, self).__init__()
        self.factor = factor

        self.deconv = lambda x: F.interpolate(
            x, scale_factor=self.factor, mode='linear', align_corners=True
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.deconv(x)
        return x.permute(0, 2, 1)

class TDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(TDecoderBlock, self).__init__()
        self.conv = nn.Sequential(
            sb.nnet.CNN.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                stride=1
            ),
            nn.GLU()
        )

        self.upsampling = nn.Sequential(
            Upsampling(factor=stride),
            nn.GELU()
        )
    
    def forward(self, x, x_skip):
        # Residual skip connection
        x = x + x_skip

        # Decoder blocks
        x = self.conv(x)
        x = self.upsampling(x)
        return x


class TDecoder(nn.Module):
    def __init__(
        self, 
        base_ch, 
        kernel_size, 
        stride, 
        depth
    ):
        super(TDecoder, self).__init__()
        self.depth = depth
        self.TDec = nn.ModuleList()

        # Decoder
        for d in range(self.depth, 0, -1):
            chnn = (2**(d-1))*base_ch
            self.TDec.append(
                TDecoderBlock(chnn, chnn, kernel_size, stride)
            )

        self.conv = sb.nnet.CNN.Conv1d(
            in_channels=base_ch//2,
            out_channels=1,
            kernel_size=1,
            stride=1
        )
    
    def forward(self, x, x_skip):
        for layer in range(self.depth):
            skip_conn = x_skip.pop()
            x = self.TDec[layer](x, skip_conn)

        return self.conv(x)