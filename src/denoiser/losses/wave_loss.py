import torch
import torch.nn as nn
from torch_stoi import NegSTOILoss
from speechbrain.nnet.loss.si_snr_loss import si_snr_loss

class WaveLoss(nn.Module):
    """
    Implementation of Waveform Losses. 
    Computes the distance between two raw waveforms.
    
    Arguments
    ---------
    distance_metric: str
        Which distance metric to use it (L1, L2 or STOI)
    sample_rate: int
        Audio sample rate
    use_log: bool
        Wheter to use logarithmically-scaled loss
    """
    def __init__(
        self, 
        distance_metric:str="stoi",
        sample_rate:int=16000,
        use_log:bool=False
    ):
        super().__init__()
        self.distance_metric = distance_metric
        self.sample_rate = sample_rate
        self.use_log = use_log

        # Validate the passed parameters
        dist_types = ['l1', 'l2', 'stoi']
        
        if self.distance_metric not in dist_types:
            raise ValueError(
                f'distance_metric must be one of {dist_types}'
            )

        self.dist_metric = self.load_distance_metric()

    def forward(self, y_hat, y, lens=None):
        # Unfold predictions
        y_hat, y = y_hat.squeeze(-1), y.squeeze(-1)
        loss = self.dist_metric(y_hat, y)
        
        if self.use_log and (self.distance_metric != 'stoi'):
            loss = 10*torch.log10(loss)

        return loss.mean()

    def load_distance_metric(self):
        fns = {
            'l1': nn.L1Loss(),
            'l2': nn.MSELoss(),
            'stoi': NegSTOILoss(sample_rate=self.sample_rate)
        }

        return fns[self.distance_metric]


class SISNR(nn.Module):
    """
    Implementation of Scale-Independent Signal-to-Noise Ratio Loss
    """
    def __init__(self):
        super().__init__()
        self.dist_metric = si_snr_loss

    def forward(self, y_hat, y, lens=None):
        # Unfold predictions
        y_hat, y = y_hat.squeeze(-1), y.squeeze(-1)

        return self.dist_metric(y_hat, y, lens, reduction="mean")
