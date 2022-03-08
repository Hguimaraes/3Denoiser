import torch
import torch.nn as nn
from speechbrain.processing.features import spectral_magnitude

class STFTMagnitudeLoss(nn.Module):
    """
    Implementation of Spectral Magnitude Loss. 
    Computes the distance between two STFT representations.
    
    Arguments
    ---------
    distance_metric: str
        Which distance metric to use it (L1 or L2)
    compute_stft: object
        Function to compute the STFT representations.
    use_log: bool
        Wheter to use logarithmically-scaled loss
    """
    def __init__(
        self, 
        distance_metric:str="l2",
        compute_stft:object=None,
        use_log:bool=False
    ):
        super().__init__()
        self.distance_metric = distance_metric
        self.compute_stft = compute_stft
        self.use_log = use_log

        # Validate the passed parameters
        dist_types = ['l1', 'l2']
        
        if self.distance_metric not in dist_types:
            raise ValueError(
                f'distance_metric must be one of {dist_types}'
            )

        self.dist_metric = self.load_distance_metric()

    def forward(self, y_hat, y, lens=None):
        # Unfold predictions
        y_hat, y = y_hat.squeeze(-1), y.squeeze(-1)

        # Compute STFT representations
        rep_y_hat, rep_y = map(self.compute_features, [y_hat, y])
        loss = self.dist_metric(rep_y_hat, rep_y)

        if self.use_log:
            loss = 10*torch.log10(loss)

        return loss.mean()
    
    def load_distance_metric(self):
        fns = {
            'l1': nn.L1Loss(),
            'l2': nn.MSELoss()
        }

        return fns[self.distance_metric]
    
    def compute_features(self, x):
        # Spectrogram
        feats = self.compute_stft(x)
        feats = spectral_magnitude(feats, power=0.5)

        return feats

class LogSTFTMagnitudeLoss(nn.Module):
    """
    Implementation of Log Spectral Magnitude Loss. 
    Computes the distance between two Log-STFT representations.
    
    Arguments
    ---------
    distance_metric: str
        Which distance metric to use it (L1 or L2)
    compute_stft: object
        Function to compute the STFT representations.
    """
    def __init__(
        self, 
        distance_metric:str="l2",
        compute_stft:object=None,
        use_log:bool=False
    ):
        super().__init__()
        self.distance_metric = distance_metric
        self.compute_stft = compute_stft
        self.use_log = use_log

        # Validate the passed parameters
        dist_types = ['l1', 'l2']
        
        if self.distance_metric not in dist_types:
            raise ValueError(
                f'distance_metric must be one of {dist_types}'
            )

        self.dist_metric = self.load_distance_metric()

    def forward(self, y_hat, y, lens=None):
        # Unfold predictions
        y_hat, y = y_hat.squeeze(-1), y.squeeze(-1)

        # Compute STFT representations
        rep_y_hat, rep_y = map(self.compute_features, [y_hat, y])
        loss = self.dist_metric(
            torch.log(rep_y_hat), 
            torch.log(rep_y)
        )

        return loss.mean()

    def load_distance_metric(self):
        fns = {
            'l1': nn.L1Loss(),
            'l2': nn.MSELoss()
        }

        return fns[self.distance_metric]

    def compute_features(self, x):
        # Spectrogram
        feats = self.compute_stft(x)
        feats = spectral_magnitude(feats, power=0.5)

        return feats