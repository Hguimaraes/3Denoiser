import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss
from torch_stoi import NegSTOILoss
from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
from speechbrain.lobes.models.fairseq_wav2vec import FairseqWav2Vec1

from denoiser.losses.spectral_loss import STFTMagnitudeLoss


class PFPL(nn.Module):
    """
    Implementation of Perceptual Losses for SE
    Based on the Phone-fortified Perceptual Loss
    Paper: https://arxiv.org/pdf/2010.15174v3.pdf
    Code: https://github.com/aleXiehta/PhoneFortifiedPerceptualLoss
    
    Arguments
    ---------
    PRETRAINED_MODEL_PATH: str
        Path where the weights of the model are stored
    """
    def __init__(
        self,
        PRETRAINED_MODEL_PATH:str
    ):
        super().__init__()
        self.PRETRAINED_MODEL_PATH = PRETRAINED_MODEL_PATH

        self.model = self.load_fairseq()
        self.dist_metric = SamplesLoss()

    def forward(self, y_hat, y, lens=None):
        # Unfold predictions
        y_hat, y = y_hat.squeeze(-1), y.squeeze(-1)

        # Compute latent representations
        with torch.no_grad():
            rep_y_hat, rep_y = map(self.model, [y_hat, y])

        wass_dist = self.dist_metric(rep_y_hat, rep_y)

        return wass_dist.mean() + F.l1_loss(y_hat, y).mean()
    

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
        model.eval()

        return model.to("cuda")


class DeepFeatureLoss(nn.Module):
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
    """
    def __init__(self, 
        PRETRAINED_MODEL_PATH:str,
        alpha:float=.5,
        model_architecture:str='wav2vec2',
        compute_stft:object=None
    ):
        super().__init__()
        self.alpha = alpha
        self.PRETRAINED_MODEL_PATH = PRETRAINED_MODEL_PATH
        self.model_architecture = model_architecture

        # Validate the passed parameters
        model_types = ['wav2vec', 'wav2vec2', 'hubert']

        if self.model_architecture not in model_types:
            raise ValueError(
                f'model_architecture must be one of {model_types}'
            )

        self.model = self.load_representation_model()
        self.dist_metric = nn.MSELoss()
        self.reconstruction_loss = STFTMagnitudeLoss(
            distance_metric="l2", 
            compute_stft=compute_stft
        )

    def forward(self, y_hat, y, lens=None):
        # STFT magnitude loss
        l2_loss = self.reconstruction_loss(y_hat, y)

        # Unfold predictions
        y_hat, y = y_hat.squeeze(-1), y.squeeze(-1)

        # Compute latent representations
        with torch.no_grad():
            rep_y_hat, rep_y = map(self.model, [y_hat, y])

        dist = self.dist_metric(rep_y_hat, rep_y).mean()

        return l2_loss + self.alpha*dist
    

    def load_representation_model(self):
        models_table = {
            "wav2vec": self.load_fairseq,
            "wav2vec2": self.load_speechbrain,
            "hubert": self.load_speechbrain
        }

        model = models_table[self.model_architecture]()
        model.eval()

        return model#.to("cuda")

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


class CompoundedPerceptualLoss(nn.Module):
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
        alpha:float=10,
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
                f'distance_metric must be one of {distance_metric}'
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


class sbSRLWrapper(nn.Module):
    """
    """
    def __init__(
        self,
        upstream:object=None,
        n_layers:int=7
    ):
        super().__init__()
        self.model = upstream.model.feature_extractor
        
        self.features = {}
        for n in range(n_layers):
            self.model.conv_layers[n].register_forward_hook(
                self.get_features(f'layer_{n}')
            )

    def forward(self, x):
        out = self.model(x)
        return self.features
    
    def get_features(self, name):
        def hook(model, input, output):
            self.features[name] = output
        return hook