import speechbrain as sb

import os
import torch
import numpy as np
import speechbrain as sb
from tqdm.contrib import tqdm
from torch.utils.data import DataLoader

class DenoiserBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        noisy_wavs, lens = batch.predictor # B, C, L

        return self.modules.model(noisy_wavs)
    
    def compute_objectives(self, predictions, batch, stage):
        # Get clean targets
        targets, lens = batch.target

        # Compare the waveforms
        loss = self.modules.loss(predictions, targets, lens)

        return loss