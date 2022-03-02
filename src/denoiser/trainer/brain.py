import speechbrain as sb

import os
import torch
import logging
import numpy as np
import speechbrain as sb
from tqdm.contrib import tqdm
from torch.utils.data import DataLoader
from denoiser.losses import task1_metric
import speechbrain.nnet.schedulers as schedulers


# Logger info
logger = logging.getLogger(__name__)

class DenoiserBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        noisy_wavs, lens = batch.predictor # B, C, L

        return self.modules.model(noisy_wavs)
    
    def compute_objectives(self, predictions, batch, stage):
        # Get clean targets
        targets, lens = batch.target # B, L, C

        # Compare the waveforms
        loss = self.modules.loss(predictions, targets, lens)

        # Since it is slow, only compute task1 metric on evalution sets
        if (stage != sb.Stage.TRAIN):
            self.l3das_task1_metric.append(
                batch.id,
                np.squeeze(targets.cpu().numpy()),
                np.squeeze(predictions.cpu().numpy())
            )

        return loss
    
    def on_stage_start(self, stage, epoch=None):
        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=self.modules.loss
        )

        # Add a metric for evaluation sets
        if stage != sb.Stage.TRAIN:
            self.l3das_task1_metric = sb.utils.metric_stats.MetricStats(
                metric=task1_metric
            )

    def on_stage_end(self, stage, stage_loss, epoch=None):
        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # At the end of validation, we can write stats and checkpoints
        if (stage == sb.Stage.VALID):
            # Summarize the statistics from the stage for record-keeping.
            stats = {
                "loss": stage_loss,
                "task1_metric": self.l3das_task1_metric.summarize("average"),
            }

            current_lr, next_lr = self.hparams.lr_scheduler(
                [self.optimizer], epoch, stage_loss
            )
            schedulers.update_learning_rate(self.optimizer, next_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch": epoch, "LR": current_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            # unless they have the current best task1_metric
            self.checkpointer.save_and_keep_only(meta=stats, max_keys=["task1_metric"])
    
    def fit_batch(self, batch):
        """Trains the parameters given a single batch in input"""

        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        if self.hparams.threshold_byloss:
            th = self.hparams.threshold
            loss_to_keep = loss[loss > th]
            if loss_to_keep.nelement() > 0:
                loss = loss_to_keep.mean()
        else:
            loss = loss.mean()
        
         # the fix for computational problems
        if (loss < self.hparams.loss_upper_lim and loss.nelement() > 0):
            # normalize the loss by gradient_accumulation step
            (loss / self.hparams.gradient_accumulation).backward()

            if self.step % self.hparams.gradient_accumulation == 0:
                if self.hparams.clip_grad_norm >= 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.modules.parameters(), self.hparams.clip_grad_norm
                    )

                self.check_gradients(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
        else:
            self.nonfinite_count += 1
            logger.info(
                "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                    self.nonfinite_count
                )
            )
            loss.data = torch.tensor(0).to(self.device)

        return loss.detach()

    def predict(
        self,
        test_set,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={},
    ):
        if progressbar is None:
            progressbar = not self.noprogressbar
        
        # Construct test data loader
        if not isinstance(test_set, DataLoader):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, sb.Stage.TEST, **test_loader_kwargs
            )
        
        # Call the predict
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()

        with torch.no_grad():
            for batch in tqdm(
                test_set, dynamic_ncols=True, disable=not progressbar
            ):
                self.step += 1
                out = self.compute_forward(batch, stage=sb.Stage.TEST)
                self.write_audios(batch.id, batch.length, out)
        self.step = 0
    
    def write_audios(self, utt_id, utt_length, batch):
        audio_path = self.hparams.audio_result
        npy_path = self.hparams.npy_result
        if not os.path.exists(audio_path):
            os.mkdir(audio_path)
        
        if not os.path.exists(npy_path):
            os.mkdir(npy_path)

        for count, (utt, length) in enumerate(zip(utt_id, utt_length)):
            audio = batch[count, :length, 0].detach().cpu()
            # Save as wav file to listen to the result
            sb.dataio.dataio.write_audio(
                filepath=os.path.join(
                    audio_path, 
                    "{}.wav".format(utt)
                ),
                audio=audio/audio.max()*0.9, # Normalization
                samplerate=16000
            )

            # Save as .npy for submission
            with open(
                os.path.join(npy_path, 
                "{}.npy".format(utt)), 'wb'
            ) as f:
                np.save(f, audio.numpy())