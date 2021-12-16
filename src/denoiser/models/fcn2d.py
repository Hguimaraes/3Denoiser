import julius
import torch
from copy import copy
import torch.nn as nn
import speechbrain as sb
from speechbrain.processing.features import spectral_magnitude
from typing import Tuple

class FCN2D(nn.Module):
    def __init__(
        self,
        rep_channels:int=64,
        kernel_size:Tuple[int, int]=(9, 9),
        compute_stft:object=None,
        resynth:object=None,
        resampling:bool=True
    ):
        super(FCN2D, self).__init__()
        self.rep_channels=rep_channels
        self.kernel_size=kernel_size
        self.compute_stft=compute_stft
        self.resynth=resynth
        self.resampling=resampling

        self.nnet_layers = sb.nnet.containers.Sequential(
            self.conv_block(1, self.rep_channels),
            self.conv_block(self.rep_channels, self.rep_channels),
            self.conv_block(self.rep_channels, self.rep_channels),
            self.conv_block(self.rep_channels, self.rep_channels),
            self.conv_block(self.rep_channels, self.rep_channels),
            self.conv_block(self.rep_channels, self.rep_channels),
            self.conv_block(self.rep_channels, self.rep_channels),
            sb.nnet.CNN.Conv2d(
                in_channels=self.rep_channels,
                out_channels=1,
                kernel_size=self.kernel_size
            ),
            nn.Sigmoid()
        )

    def forward(self, noisy_wavs):
        # Select W channel from both microphones
        noisy_wavs = noisy_wavs[:, :, 0]
        x = copy(noisy_wavs)

        # Resample to avoid aliasing artifacts
        if self.resampling:
            x = self.resample(x, 1, 2)

        # Extract features
        noisy_spec = self.compute_features(x)
        noisy_spec = noisy_spec.unsqueeze(-1)
        length = noisy_spec.shape[1]

        # Encoders network for representation learning
        mask = self.nnet_layers(noisy_spec)
        predict_spec = torch.mul(mask, noisy_spec)

        # Fix sizes from transposed conv
        predict_spec = predict_spec[:, :length, :].squeeze(-1)

        # resynth the time-frequency representation
        x_hat = self.resynth(torch.expm1(predict_spec), noisy_wavs)
        x_hat = x_hat.unsqueeze(2)

        # Return the sampled back audio
        if self.resampling:
            x = self.resample(x, 2, 1)

        return x_hat
    
    """
    Extract spectrogram and manipulate the waveform
    """
    def compute_features(self, x):
        # Spectrogram
        feats = self.compute_stft(x)
        feats = spectral_magnitude(feats, power=0.5)
        feats = torch.log1p(feats)

        return feats
    
    def resample(self, x, from_sample, to_sample):
        x = x.transpose(1, 2) # B, L, C => B, C, L
        x = julius.resample_frac(x, from_sample, to_sample)

        return x.transpose(1, 2) # B, C, L => B, L, C
    
    def conv_block(self, in_channels, base_channels):
        return sb.nnet.containers.Sequential(
            sb.nnet.CNN.Conv2d(
                in_channels=in_channels,
                out_channels=base_channels,
                kernel_size=self.kernel_size,
                padding="same"
            ),
            sb.nnet.normalization.BatchNorm2d(input_size=base_channels),
            nn.LeakyReLU(0.01)
        )