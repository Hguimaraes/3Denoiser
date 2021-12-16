import julius
import torch
from copy import copy
import torch.nn as nn
import torch.nn.functional as F
import speechbrain as sb
from speechbrain.processing.features import spectral_magnitude
from speechbrain.lobes.beamform_multimic import DelaySum_Beamformer

from denoiser.models import MaskNet

class STFTransformer(nn.Module):
    def __init__(
        self,
        n_fft:int=16,
        attn_base_ch:int=512,
        dropout:float=0.1,
        num_layers:int=8,
        d_ffn:int=512,
        nhead:int=12,
        causal:bool=True,
        compute_stft:object=None,
        resynth:object=None,
        resampling:bool=True,
        embedding_type:str="cnntransformer",
        sampling_rate:int=16000
    ):
        super(STFTransformer, self).__init__()
        self.n_fft = n_fft
        self.attn_base_ch = attn_base_ch
        self.dropout = dropout
        self.num_layers = num_layers
        self.d_ffn = d_ffn
        self.nhead = nhead
        self.causal = causal
        self.compute_stft=compute_stft
        self.resynth=resynth
        self.resampling=resampling
        self.embedding_type = embedding_type
        self.sampling_rate = sampling_rate

        if self.embedding_type not in ["conformer", "cnntransformer"]:
            raise ValueError("Invalid option for embedding_type paramter")
        
        self.beamformer = DelaySum_Beamformer(sampling_rate=self.sampling_rate)

        if self.embedding_type == "cnntransformer":
            self.transformer = MaskNet(
                in_channels=self.n_fft // 2 + 1, 
                base_channels=self.attn_base_ch,
                dropout=self.dropout,
                num_layers=self.num_layers,
                d_ffn=self.d_ffn,
                nhead=self.nhead,
                causal=self.causal
            )
        else:
            raise NotImplementedError()

    def forward(self, noisy_wavs):
        # Select W channel from both microphones
        length = noisy_wavs.shape[1]
        noisy_wavs = noisy_wavs[:, :, (0, 4)].mean(dim=-1)
        # noisy_wavs = self.beamformer(noisy_wavs).squeeze(-1)
        x = copy(noisy_wavs)

        # Resample to avoid aliasing artifacts
        if self.resampling:
            x = self.resample(x, 1, 2)

        # Extract features
        noisy_spec = self.compute_features(x)

        # Encoders network for representation learning
        mask = self.transformer(noisy_spec)
        predict_spec = torch.mul(mask, noisy_spec)

        # Fix sizes from transposed conv
        predict_spec = predict_spec.squeeze(-1)

        # resynth the time-frequency representation
        x_hat = self.resynth(torch.expm1(predict_spec), noisy_wavs)
        x_hat = F.pad(x_hat, (length - x_hat.shape[1], 0)).unsqueeze(-1)

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