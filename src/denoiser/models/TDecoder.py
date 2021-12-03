import torch
import torch.nn as nn
import torch.nn.functional as F
import speechbrain as sb

class TDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(TDecoderBlock, self).__init__()
        self.conv_1x1 = nn.Sequential(
            sb.nnet.CNN.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1
            ),
            nn.GLU()
        )
        self.trans_conv = nn.Sequential(
            sb.nnet.CNN.ConvTranspose1d(
                in_channels=in_channels//2,
                out_channels=out_channels//2,
                kernel_size=kernel_size,
                stride=stride,
                padding=2
            ),
            sb.nnet.normalization.InstanceNorm1d(
                input_size=out_channels//2,
                track_running_stats=False,
                affine=True
            ),
            nn.GELU()
        )
    
    def forward(self, x, x_skip):
        # Residual skip connection
        x = x + x_skip

        # Decoder blocks
        x = self.conv_1x1(x)
        x = self.trans_conv(x)
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