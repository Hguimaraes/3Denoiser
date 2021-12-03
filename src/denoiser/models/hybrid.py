import julius
import torch.nn as nn

from denoiser.models import TEncoder
from denoiser.models import TDecoder
from denoiser.models import Bootleneck
from denoiser.utils import pad_power

from speechbrain.processing.features import STFT, ISTFT
from speechbrain.lobes.beamform_multimic import DelaySum_Beamformer

class HybridDenoiser(nn.Module):
    def __init__(
        self, 
        in_ch:int=1, 
        depth:int=5,
        base_ch:int=64,
        T_ks:int=8,
        T_stride:int=4,
        sample_rate:int=16000
    ):
        super(HybridDenoiser, self).__init__()
        self.depth = depth
        self.TEnc = TEncoder(
            in_ch=in_ch,
            base_ch=base_ch,
            kernel_size=T_ks,
            stride=T_stride,
            depth=depth
        )

        self.bootleneck = Bootleneck(base_ch=base_ch, depth=depth)

        self.TDec = TDecoder(
            base_ch=base_ch,
            kernel_size=T_ks,
            stride=T_stride,
            depth=depth
        )

        # Feature functions
        self.stft = STFT(sample_rate=sample_rate)
        self.istft = ISTFT(sample_rate=sample_rate)
        self.beamformer = DelaySum_Beamformer(sampling_rate=sample_rate)

    def forward(self, x):
        # Resample to avoid aliasing artifacts
        x = julius.resample_frac(x, 1, 2)

        # Speechbrain format
        x = x.transpose(1, 2) # B, C, L => B, L, C

        # Extract features
        # x, x_spec = self.compute_features(x)

        # Time-Domain network
        x, x_skip = self.TEnc(x[:, :, 0].unsqueeze(2))
        x = self.bootleneck(x)
        x = self.TDec(x, x_skip)
        
        x = x.transpose(1, 2) # B, L, C => B, C, L

        # Return the sampled back audio
        return julius.resample_frac(x, 2, 1)
    
    def compute_features(self, x):
        n = x.shape[1]

        # Spectrogram
        # Xs = self.stft(x)
        Xs = None

        # Apply Delay and Sum Beamforming
        x = self.beamformer(x)
        x = nn.functional.pad(x, (0, 0, 0, n - x.shape[1]), "constant", 0)
        
        return x, Xs