import torch.nn as nn
import torch.nn.functional as F
import speechbrain as sb

class TEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(TEncoderBlock, self).__init__()
        self.conv = nn.Sequential(
            # GELU(Conv1d(x))
            sb.nnet.CNN.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding='causal'
            ),
            nn.GELU(),

            # GLU(Conv1d(x))
            sb.nnet.CNN.Conv1d(
                in_channels=out_channels,
                out_channels=2*out_channels,
                kernel_size=1,
                stride=1
            ),
            nn.GLU()
        )

    def forward(self, x):
        return self.conv(x)


class TEncoder(nn.Module):
    def __init__(
        self, 
        in_ch, 
        base_ch, 
        kernel_size, 
        stride, 
        depth
    ):
        super(TEncoder, self).__init__()
        self.TEnc = nn.ModuleList()
        self.TEnc.append(
            TEncoderBlock(in_ch, base_ch, kernel_size, stride)
        )

        # Encoder
        for d in range(depth-1):
            chnn = (2**d)*base_ch
            self.TEnc.append(
                TEncoderBlock(chnn, 2*chnn, kernel_size, stride)
            )
    
    def forward(self, x):
        x_skip = []
        for l in self.TEnc:
            x = l(x)
            x_skip.append(x)
        return x, x_skip