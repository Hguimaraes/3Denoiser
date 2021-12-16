from copy import copy

import julius
import torch
import torch.nn as nn
import speechbrain as sb
from speechbrain.processing.features import spectral_magnitude

from denoiser.models import MaskNet


class HybridDenoiser(nn.Module):
    def __init__(
        self, 
        in_channels:int=1,
        rep_channels:int=256,
        T_kernel_size:int=15,
        T_stride:int=8,
        Z_kernel_size:int=5,
        Z_stride:int=2,
        attn_base_ch:int=512,
        dropout:float=0.1,
        num_layers:int=8,
        d_ffn:int=512,
        nhead:int=12,
        causal:bool=True,
        sample_rate:int=16000,
        compute_stft:object=None,
        resynth:object=None,
        resampling:bool=True,
        n_fft:int=512
    ):
        super(HybridDenoiser, self).__init__()

        # Save parameters
        self.in_channels=in_channels
        self.rep_channels=rep_channels
        self.T_kernel_size=T_kernel_size
        self.T_stride=T_stride
        self.attn_base_ch=attn_base_ch
        self.dropout=dropout
        self.num_layers=num_layers
        self.d_ffn=d_ffn
        self.nhead=nhead
        self.causal=causal
        self.sample_rate=sample_rate
        self.compute_stft=compute_stft
        self.resynth=resynth
        self.resampling=resampling
        self.Z_kernel_size=Z_kernel_size
        self.Z_stride=Z_stride
        self.n_fft=n_fft

        # Time-Domain Network
        self.TEnc = sb.nnet.containers.Sequential(
            sb.nnet.CNN.SincConv(
                in_channels=self.in_channels,
                out_channels=16,
                kernel_size=11,
                stride=1
            ),
            nn.ReLU(),

            sb.nnet.CNN.Conv1d(
                in_channels=16,
                out_channels=self.rep_channels,
                kernel_size=self.T_kernel_size,
                stride=self.T_stride
            ),
            nn.ReLU()
        )

        self.TDec = sb.nnet.containers.Sequential(
            sb.nnet.CNN.ConvTranspose1d(
                in_channels=self.rep_channels,
                out_channels=1,
                kernel_size=self.T_kernel_size,
                stride=self.T_stride
            )
        )

        # Shared bootleneck
        self.bootleneck = MaskNet(
            in_channels=self.rep_channels, 
            base_channels=self.attn_base_ch,
            dropout=self.dropout,
            num_layers=self.num_layers,
            d_ffn=self.d_ffn,
            nhead=self.nhead,
            causal=self.causal
        )

    def forward(self, noisy_wavs):
        # Select W channel from both microphones
        noisy_wavs = noisy_wavs[:, :, 0].unsqueeze(dim=2)
        x = copy(noisy_wavs)

        # Resample to avoid aliasing artifacts
        if self.resampling:
            x = self.resample(x, 1, 2)

        # Extract features
        z = self.compute_features(x)
        t_length = x.shape[1]
        z_length = z.shape[1]

        # Encoders network for representation learning
        x = self.TEnc(x)

        # Bootleneck
        mask_x, mask_z = self.bootleneck(x, z)
        z = torch.mul(mask_z, z)
        x = torch.mul(mask_x, x)

        # Decoders network
        x = self.TDec(x)

        # Fix sizes from transposed conv
        x = x[:, :t_length, :]
        z = z[:, :z_length, :]

        # resynth the time-frequency representation
        x_spec = self.resynth(torch.expm1(z), noisy_wavs.squeeze(2))
        x_spec = x_spec.unsqueeze(2)

        # Return the sampled back audio
        if self.resampling:
            x = self.resample(x, 2, 1)

        return x + x_spec
    
    """
    Extract spectrogram and manipulate the waveform
    """
    def compute_features(self, x):
        # B, L, C ==> B, L
        x = x.squeeze(2)

        # Spectrogram
        feats = self.compute_stft(x)
        feats = spectral_magnitude(feats, power=0.5)
        feats = torch.log1p(feats)

        return feats
    
    def resample(self, x, from_sample, to_sample):
        x = x.transpose(1, 2) # B, L, C => B, C, L
        x = julius.resample_frac(x, from_sample, to_sample)

        return x.transpose(1, 2) # B, C, L => B, L, C