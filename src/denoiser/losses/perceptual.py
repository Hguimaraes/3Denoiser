import torch
import torch.nn as nn
from geomloss import SamplesLoss
from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
from torch_stoi import NegSTOILoss


# Perceptual Loss
# Paper: https://arxiv.org/pdf/2010.15174v3.pdf
# https://github.com/aleXiehta/PhoneFortifiedPerceptualLoss
class PerceptualLoss(nn.Module):
    def __init__(self, PRETRAINED_MODEL_PATH:str, alpha:float=.5):
        super().__init__()
        self.alpha = alpha
        self.stoi = NegSTOILoss(sample_rate=16000)

        self.model = HuggingFaceWav2Vec2(
            "facebook/wav2vec2-base-960h",
            save_path=PRETRAINED_MODEL_PATH
        )
        self.model = self.model.to("cuda:0")
        self.model.eval()
        self.dist_metric = nn.MSELoss()

    def forward(self, y_hat, y, lens=None):
        # Unfold predictions
        y_hat, y = y_hat.squeeze(-1), y.squeeze(-1)
        stoi_loss = self.stoi(y_hat, y)

        # Compute latent representations
        with torch.no_grad():
            rep_y_hat, rep_y = map(self.model, [y_hat, y])

        return self.alpha*self.dist_metric(rep_y_hat, rep_y).mean() + stoi_loss.mean()