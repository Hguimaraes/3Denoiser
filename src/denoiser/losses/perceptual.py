import torch
import torch.nn as nn
from torch_stoi import NegSTOILoss
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

        # self.model = HuggingFaceWav2Vec2("facebook/wav2vec2-base-960h", save_path="../models/")
        # self.model = self.model.to("cuda:0")
        # self.model.eval()

        self.dist_metric = nn.L1Loss()

    def forward(self, y_hat, y, lens=None):
        # Unfold predictions
        y_hat, y = y_hat.squeeze(2), y.squeeze(2)

        fn_loss = stoi_loss(y_hat, y, lens=lens)
        sc_loss, mag_loss = self.stft_loss(y_hat, y)
        # rep_y_hat, rep_y = map(self.model, [y_hat.squeeze(1), y.squeeze(1)])
        # return self.alpha*self.dist_metric(rep_y_hat, rep_y).mean() + stoi_loss.mean()
        return fn_loss + sc_loss + mag_loss