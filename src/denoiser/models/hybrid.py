import julius
import torch
import torch.nn as nn

from denoiser.models import TEncoder
from denoiser.models import TDecoder
from denoiser.models import Bootleneck

from speechbrain.processing.features import STFT, ISTFT
from speechbrain.processing.features import spectral_magnitude

class HybridDenoiser(nn.Module):
    def __init__(
        self, 
        in_ch:int=1, 
        depth:int=5,
        base_ch:int=64,
        T_ks:int=8,
        T_stride:int=4,
        sample_rate:int=16000,
        normalize:bool=True,
        floor:float=1e-3,
        resampling:bool=True,
        compute_stft:object=None,
        compute_istft:object=None
    ):
        super(HybridDenoiser, self).__init__()
        self.depth = depth
        self.normalize = normalize
        self.floor = floor
        self.stride = T_stride
        self.kernel_size = T_ks
        self.resampling = resampling
        self.compute_stft = compute_stft
        self.compute_istft = compute_istft

        self.TEnc = TEncoder(
            in_ch=in_ch,
            base_ch=base_ch,
            kernel_size=self.kernel_size,
            stride=self.stride,
            depth=depth
        )

        self.bootleneck = Bootleneck(base_ch=base_ch, depth=depth)

        self.TDec = TDecoder(
            base_ch=base_ch,
            kernel_size=self.kernel_size,
            stride=self.stride,
            depth=depth
        )

        # Feature functions
        self.stft = STFT(sample_rate=sample_rate)
        self.istft = ISTFT(sample_rate=sample_rate)

    def forward(self, x):
        # Select W channel from both microphones
        x = x[:, :, (0, 4)]

        if self.normalize:
            mono = x.mean(dim=2, keepdim=True)
            std = mono.std(dim=1, keepdim=True)
            x = x / (self.floor + std)
        else:
            std = 1

        # Resample to avoid aliasing artifacts
        if self.resampling:
            x = self.resample(x, 1, 2)

        # Extract features
        length = x.shape[1]
        x, x_spec = self.compute_features(x)

        # Time-Domain network
        x, x_skip = self.TEnc(x)
        x = self.bootleneck(x)
        x = self.TDec(x, x_skip)

        # Remove pad
        x = x[:, :length, :]

        # Return the sampled back audio
        if self.resampling:
            x = self.resample(x, 2, 1)
        
        return x * std
    
    """
    Extract spectrogram and manipulate the waveform
    """
    def compute_features(self, x):
        x = self.pad_power(x)

        # Spectrogram
        feats = self.compute_stft(x)
        feats = spectral_magnitude(feats, power=0.5)
        feats = torch.log1p(feats)

        return x, feats

    """
    Pad an audio to a power of 4**depth
    necessary for decimate operation in the network
    """
    def pad_power(self, batch: torch.Tensor):
        power = self.stride**self.depth-1
        length = batch.shape[1]
        batch = batch.transpose(1, 2)

        if length % (power + 1) != 0:
            diff = (length|power) + 1 - length
            batch = torch.nn.functional.pad(batch, (0, diff))

        return batch.transpose(1, 2)
    
    def resample(self, x, from_sample, to_sample):
        x = x.transpose(1, 2) # B, L, C => B, C, L
        x = julius.resample_frac(x, from_sample, to_sample)

        return x.transpose(1, 2) # B, C, L => B, L, C