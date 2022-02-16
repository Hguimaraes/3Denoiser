import julius
import torch
from copy import copy
import torch.nn as nn
from typing import Tuple
import speechbrain as sb
from speechbrain.nnet.complex_networks.c_CNN import CConv2d
from speechbrain.processing.features import spectral_magnitude
from speechbrain.nnet.complex_networks.c_normalization import CBatchNorm


class TFFCN2D(nn.Module):
    """This function implements a Fully Convolutional Network on STFT (mag).
    Arguments
    ---------
    num_channels : int
        Number of input channels.
    rep_channels : int
        Number of channels in the intermediate channels.
    kernel_size: tuple
        Kernel size of the convolutional filters.
    compute_stft : object
        Function to compute the STFT representations.
    resynth : object
        Method to compute the inverse STFT and reconstruct the audio.
    resampling : bool
        Use or not a resampling method to double the sampling rate.
    """
    def __init__(
        self,
        num_channels:int=4,
        rep_channels:int=64,
        kernel_size:Tuple[int, int]=(9, 9),
        compute_stft:object=None,
        resynth:object=None,
        resampling:bool=True
    ):
        super(TFFCN2D, self).__init__()
        self.num_channels=num_channels
        self.rep_channels=rep_channels
        self.kernel_size=kernel_size
        self.compute_stft=compute_stft
        self.resynth=resynth
        self.resampling=resampling

        self.nnet_layers = sb.nnet.containers.Sequential(
            self.conv_block(self.num_channels, self.rep_channels),
            self.conv_block(self.rep_channels, self.rep_channels),
            self.conv_block(self.rep_channels, self.rep_channels),
            self.conv_block(self.rep_channels, self.rep_channels),
            sb.nnet.CNN.Conv2d(
                input_shape=(None, None, None, self.rep_channels),
                out_channels=self.num_channels,
                kernel_size=self.kernel_size
            ),
            nn.Sigmoid()
        )

        self.output_layer = sb.nnet.CNN.Conv2d(
            input_shape=(None, None, None, self.num_channels),
            out_channels=1,
            kernel_size=self.kernel_size,
            padding="same"
        )

    def forward(self, noisy_wavs):
        # Select W channel from both microphones
        noisy_wavs = noisy_wavs[:, :, 0:self.num_channels]
        length = noisy_wavs.shape[1]
        x = copy(noisy_wavs)

        # Resample to avoid aliasing artifacts
        if self.resampling:
            x = self.resample(x, 1, 2)

        # Extract features
        noisy_spec = torch.stack([
            self.compute_features(wave) 
                for wave in torch.unbind(x, dim=-1)
        ], dim=-1)

        # Mask prediction and mono-channel
        mask = self.nnet_layers(noisy_spec)
        predict_spec = torch.mul(mask, noisy_spec)
        predict_spec = self.output_layer(predict_spec)
        predict_spec = predict_spec.squeeze(-1)

        # resynth the time-frequency representation
        x_hat = self.resynth(
            torch.expm1(predict_spec), 
            noisy_inputs=x.mean(dim=-1),
        )

        # Return the sampled back audio
        if self.resampling:
            x_hat = self.resample(x_hat, 2, 1)

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


class TFFC2N2D(nn.Module):
    """This function implements a Fully Complex Convolutional Network on STFT.
    Based on the "Deep Complex Networks", Trabelsi C. et al.
    Arguments
    ---------
    num_channels : int
        Number of input channels.
    rep_channels : int
        Number of channels in the intermediate channels.
    kernel_size: tuple
        Kernel size of the convolutional filters.
    compute_stft : object
        Function to compute the STFT representations.
    resynth : object
        Method to compute the inverse STFT and reconstruct the audio.
    resampling : bool
        Use or not a resampling method to double the sampling rate.
    """
    def __init__(
        self,
        num_channels:int=4,
        rep_channels:int=64,
        kernel_size:Tuple[int, int]=(9, 9),
        compute_stft:object=None,
        resynth:object=None,
        resampling:bool=True
    ):
        super(TFFC2N2D, self).__init__()
        self.num_channels = num_channels
        self.rep_channels=rep_channels
        self.kernel_size=kernel_size
        self.compute_stft=compute_stft
        self.resynth=resynth
        self.resampling=resampling

        self.nnet_layers = sb.nnet.containers.Sequential(
            self.conv_block(self.num_channels, self.rep_channels),
            self.conv_block(self.rep_channels, self.rep_channels),
            self.conv_block(self.rep_channels, self.rep_channels),
            self.conv_block(self.rep_channels, self.rep_channels),
            CConv2d(
                input_shape=(None, None, None, 2*self.rep_channels),
                out_channels=self.num_channels,
                kernel_size=self.kernel_size
            ),
            nn.Sigmoid()
        )

        self.output_layer = CConv2d(
            input_shape=(None, None, None, 2*self.num_channels),
            out_channels=1,
            kernel_size=self.kernel_size,
            padding="same"
        )

    def forward(self, noisy_wavs):
        # Select W channel from both microphones
        noisy_wavs = noisy_wavs[:, :, 0:self.num_channels]
        length = noisy_wavs.shape[1]
        x = copy(noisy_wavs)
        
        # Resample to avoid aliasing artifacts
        if self.resampling:
            x = self.resample(x, 1, 2)

        # Extract features
        noisy_spec = self.compute_features(x)

        # Mask prediction and mono-channel
        mask = self.nnet_layers(noisy_spec)
        predict_spec = torch.mul(mask, noisy_spec)
        predict_spec = self.output_layer(predict_spec)
        predict_spec = predict_spec.unsqueeze(-1)

        # resynth the time-frequency representation
        x_hat = self.resynth(predict_spec, sig_length=length)

        # Return the sampled back audio
        if self.resampling:
            x = self.resample(x, 2, 1)

        return x_hat

    """
    Extract spectrogram and manipulate the waveform
    """
    def compute_features(self, x):
        feats = self.compute_stft(x)
        feats = feats.transpose(3, 4)

        # Separate real and imaginary parts from the STFT
        real, img = feats[..., 0], feats[..., 1]

        return torch.cat([real, img], dim=-1)
    
    def resample(self, x, from_sample, to_sample):
        x = x.transpose(1, 2) # B, L, C => B, C, L
        x = julius.resample_frac(x, from_sample, to_sample)

        return x.transpose(1, 2) # B, C, L => B, L, C
    
    def conv_block(self, in_channels, base_channels):
        return sb.nnet.containers.Sequential(
            CConv2d(
                input_shape=(None, None, None, 2*in_channels),
                out_channels=base_channels,
                kernel_size=self.kernel_size,
                padding="same"
            ),
            CBatchNorm(
                input_shape=(None, None, None, 2*base_channels)
            ),
            nn.LeakyReLU(0.01)
        )
