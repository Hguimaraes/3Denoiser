import torch
import torch.nn as nn
from geomloss import SamplesLoss
from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
from speechbrain.nnet.loss.stoi_loss import stoi_loss

from denoiser.losses.stft_loss import MultiResolutionSTFTLoss

# Perceptual Loss
# Paper: https://arxiv.org/pdf/2010.15174v3.pdf
# https://github.com/aleXiehta/PhoneFortifiedPerceptualLoss
class PerceptualLoss(nn.Module):
    def __init__(self, PRETRAINED_MODEL_PATH:str, alpha:float=1):
        super().__init__()
        self.alpha = alpha
        self.stft_loss = MultiResolutionSTFTLoss()

        self.model = HuggingFaceWav2Vec2(
            "facebook/wav2vec2-base-960h",
            save_path=PRETRAINED_MODEL_PATH, 
            freeze=True, 
            freeze_feature_extractor=True
        )
        self.model = self.model.to("cuda:0")
        self.model.eval()

        self.dist_metric = lambda x, y: torch.abs(x - y)

    def forward(self, y_hat, y, lens=None):
        # Unfold predictions
        y_hat, y = y_hat.squeeze(2), y.squeeze(2)

        # fn_loss = stoi_loss(y_hat, y, lens=lens)
        sc_loss, mag_loss = self.stft_loss(y_hat, y)
        rep_y_hat, rep_y = map(self.model, [y_hat, y])
        # return self.alpha*self.dist_metric(rep_y_hat, rep_y).mean() + fn_loss
        # return fn_loss + sc_loss + mag_loss
        return self.alpha*self.dist_metric(rep_y_hat, rep_y).mean() + sc_loss + mag_loss