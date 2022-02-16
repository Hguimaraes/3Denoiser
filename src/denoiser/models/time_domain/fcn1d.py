from copy import copy
import torch.nn as nn
import speechbrain as sb


class TDFCN1D(nn.Module):
    """
    From paper: "End-to-End Waveform Utterance Enhancement for Direct Evaluation
    Metrics Optimization by Fully Convolutional Neural Networks", TASLP, 2018
    Arguments
    ---------
    num_channels : int
        Number of input channels.
    rep_channels : int
        Number of channels in the intermediate channels.
    kernel_size: int
        Kernel size of the convolutional filters.
    """
    def __init__(
        self, 
        num_channels:int=8,
        rep_channels:int=80,
        kernel_size:int=55,
    ):
        super(TDFCN1D, self).__init__()
        self.num_channels=num_channels
        self.rep_channels=rep_channels
        self.kernel_size=kernel_size

        # Construct FCN model as sequential
        self.model = nn.Sequential(
            sb.nnet.normalization.InstanceNorm1d(
                input_size=self.num_channels,
                track_running_stats=False,
                affine=True
            ),

            # Conv blocks
            self.conv_block(self.num_channels, self.rep_channels),
            self.conv_block(self.rep_channels, self.rep_channels),
            self.conv_block(self.rep_channels, self.rep_channels),
            self.conv_block(self.rep_channels, self.rep_channels),

            # Out conv
            sb.nnet.CNN.Conv1d(
                in_channels=self.rep_channels,
                out_channels=1,
                kernel_size=self.kernel_size
            )
        )
    
    def forward(self, noisy_wavs):
        # Select num_channels from microphones
        noisy_wavs = noisy_wavs[:, :, 0:self.num_channels]
        x = copy(noisy_wavs)

        return self.model(x)

    def conv_block(self, in_channels, base_channels):
        return sb.nnet.containers.Sequential(
            sb.nnet.CNN.Conv1d(
                in_channels=in_channels,
                out_channels=base_channels,
                kernel_size=self.kernel_size
            ),
            sb.nnet.normalization.InstanceNorm1d(
                input_size=base_channels,
                track_running_stats=False,
                affine=True
            ),
            nn.LeakyReLU(0.1)
        )
