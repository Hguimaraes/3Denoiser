import os
import torch
import torch.nn as nn
from geomloss import SamplesLoss
from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
from speechbrain.lobes.models.fairseq_wav2vec import FairseqWav2Vec1
from torch_stoi import NegSTOILoss


class PerceptualLoss(nn.Module):
    """
    Implementation of Perceptual Losses for SE
    Based on the Phone-fortified Perceptual Loss
    Paper: https://arxiv.org/pdf/2010.15174v3.pdf
    Code: https://github.com/aleXiehta/PhoneFortifiedPerceptualLoss
    
    Arguments
    ---------
    PRETRAINED_MODEL_PATH: str
        Path where the weights of the model are stored
    alpha: float
        How much emphasis the loss gives on the latent-representation distance
    model_architecture: str
        Choose the model to extract representations from.
    distance_metric: str
        Which distance metric to choose to compute latent similarity
    sample_rate: int
        Sample rate of the audios

    """
    def __init__(self, 
        PRETRAINED_MODEL_PATH:str, 
        alpha:float=.5,
        model_architecture:str='wav2vec2',
        distance_metric:str='l2',
        sample_rate:int=16000
    ):
        super().__init__()
        self.alpha = alpha
        self.PRETRAINED_MODEL_PATH = PRETRAINED_MODEL_PATH
        self.distance_metric = distance_metric
        self.model_architecture = model_architecture

        # Validate the passed parameters
        model_types = ['wav2vec', 'wav2vec2', 'hubert']
        dist_types = ['l1', 'l2', 'wass', 'kld']

        if self.model_architecture not in model_types:
            raise ValueError(
                f'model_architecture must be one of {model_types}'
            )
        
        if self.distance_metric not in dist_types:
            raise ValueError(
                f'distance_metric must be one of {model_types}'
            )

        self.stoi = NegSTOILoss(sample_rate=sample_rate)
        self.model = self.load_representation_model()
        self.dist_metric = self.load_distance_metric()        

    def forward(self, y_hat, y, lens=None):
        # Unfold predictions
        y_hat, y = y_hat.squeeze(-1), y.squeeze(-1)
        stoi_loss = self.stoi(y_hat, y)

        # Compute latent representations
        with torch.no_grad():
            rep_y_hat, rep_y = map(self.model, [y_hat, y])

        return self.alpha*self.dist_metric(rep_y_hat, rep_y).mean() + stoi_loss.mean()
    

    def load_representation_model(self):
        models_table = {
            "wav2vec": self.load_fairseq,
            "wav2vec2": self.load_speechbrain,
            "hubert": self.load_speechbrain
        }

        model = models_table[self.model_architecture]()
        model.eval()

        return model.to("cuda")
    
    def load_distance_metric(self):
        fns = {
            'l1': nn.L1Loss(),
            'l2': nn.MSELoss(),
            'wass': SamplesLoss(),
            'kld': nn.KLDivLoss()
        }

        return fns[self.distance_metric]
    

    def load_fairseq(self):
        model = FairseqWav2Vec1(
            pretrained_path=os.path.join(
                self.PRETRAINED_MODEL_PATH, 
                "wav2vec_large.pt"
            ),
            save_path=os.path.join(
                self.PRETRAINED_MODEL_PATH, 
                "wav2vec.pt"
            )
        )

        return model
    
    def load_speechbrain(self):
        model_hub = {
            "wav2vec2": "facebook/wav2vec2-base-960h", 
            "hubert": "facebook/hubert-base-ls960"
        }

        return HuggingFaceWav2Vec2(
            model_hub[self.model_architecture],
            save_path=self.PRETRAINED_MODEL_PATH
        )