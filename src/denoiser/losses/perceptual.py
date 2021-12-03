import torch
import torch.nn as nn
from torch_stoi import NegSTOILoss
from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2

# Perceptual Loss
# Paper: https://arxiv.org/pdf/2010.15174v3.pdf
# https://github.com/aleXiehta/PhoneFortifiedPerceptualLoss
class PerceptualLoss(nn.Module):
    def __init__(self, PRETRAINED_MODEL_PATH:str, alpha:float=1):
        super().__init__()
        self.alpha = alpha
        self.stoi = NegSTOILoss(sample_rate=16000)
        # self.model = HuggingFaceWav2Vec2("facebook/wav2vec2-base-960h", save_path="../models/")
        # self.model = self.model.to("cuda:0")
        # self.model.eval()

        self.dist_metric = nn.L1Loss()

    def forward(self, y_hat, y, lens=None):
        stoi_loss = self.stoi(y_hat, y)
        # rep_y_hat, rep_y = map(self.model, [y_hat.squeeze(1), y.squeeze(1)])
        # return self.alpha*self.dist_metric(rep_y_hat, rep_y).mean() + stoi_loss.mean()
        return stoi_loss.mean()