import json
import librosa
import numpy as np
from glob import glob
from pesq import pesq
from tqdm import tqdm

import warnings
import torch
import jiwer
from pystoi import stoi
import transformers
from transformers import Wav2Vec2ForMaskedLM
from transformers import Wav2Vec2Tokenizer

SR = 16000 # sample_rate
ENHANCED_FOLDER = "../logs/tf_fc2n2d_5layers_10epochs_100h_wavloss_stoi/npy_results/"
CLEANED_FOLDER = "../dataset/L3DAS22_Task1_dev/labels/"

warnings.filterwarnings("ignore", category=FutureWarning)
transformers.logging.set_verbosity_error()
wer_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
wer_model = Wav2Vec2ForMaskedLM.from_pretrained("facebook/wav2vec2-base-960h")

def main(enhanced_folder, clean_folder):
    enhanced_signals = sorted(list(glob(enhanced_folder + "*.npy")))
    reference_signals = sorted(list(glob(clean_folder + "*.wav")))

    enhanced_signals = enhanced_signals[:50]
    reference_signals = reference_signals[:50]

    se_metrics = {
        'stoi': {}, 'wer': {}, 'M': {}, 'csig': {}, 
        'cbak': {}, 'covl': {}, 'pesq': {}, 'ssnr': {}
    }
    
    for idx, path_ref in enumerate(tqdm(reference_signals)):
        # Get path from the other list
        path_enh = enhanced_signals[idx]

        # Get signal name and remove file extension
        fname_ref = path_ref.split("/").pop()[:-4]
        fname_enh = path_enh.split("/").pop()[:-4]

        assert fname_ref == fname_enh

        ref_signal, sr = librosa.load(path_ref, sr=None, mono=True)
        enh_signal = np.load(path_enh)

        # PESQ
        se_metrics['pesq'][fname_ref] = pesq_eval(ref_signal, enh_signal, SR, "wb")

        # L3DAS22 metrics
        [_stoi, _wer, m] = l3das22_metric(ref_signal, enh_signal, SR)
        se_metrics['stoi'][fname_ref] = _stoi
        se_metrics['wer'][fname_ref] = _wer
        se_metrics['M'][fname_ref] = m

    

    print("STOI = {:.2f}".format(np.mean(list(se_metrics['stoi'].values()))))
    print("WER  = {:.2f}".format(np.mean(list(se_metrics['wer'].values()))))
    print("M    = {:.2f}".format(np.mean(list(se_metrics['M'].values()))))
    print("PESQ = {:.2f}".format(np.mean(list(se_metrics['pesq'].values()))))

    with open("se_metrics.json", 'w') as f:
        json.dump(se_metrics, f)


def pesq_eval(ref_signal, pred_signal, sample_rate, mode="wb"):
    return pesq(
        fs=sample_rate,
        ref=ref_signal,
        deg=pred_signal,
        mode=mode,
    )

def wer(clean_speech, denoised_speech):
    """
    computes the word error rate(WER) score for 1 single data point
    """
    def _transcription(clean_speech, denoised_speech):

        # transcribe clean audio
        input_values = wer_tokenizer(clean_speech, return_tensors="pt").input_values;
        logits = wer_model(input_values).logits;
        predicted_ids = torch.argmax(logits, dim=-1);
        transcript_clean = wer_tokenizer.batch_decode(predicted_ids)[0];

        # transcribe
        input_values = wer_tokenizer(denoised_speech, return_tensors="pt").input_values;
        logits = wer_model(input_values).logits;
        predicted_ids = torch.argmax(logits, dim=-1);
        transcript_estimate = wer_tokenizer.batch_decode(predicted_ids)[0];

        return [transcript_clean, transcript_estimate]

    transcript = _transcription(clean_speech, denoised_speech);
    try:
        wer_val = jiwer.wer(*transcript)
    except ValueError:
        wer_val = np.NAN

    return wer_val

def l3das22_metric(clean_speech, denoised_speech, sample_rate):
    _wer = wer(clean_speech, denoised_speech)
    _stoi = stoi(clean_speech, denoised_speech, sample_rate, extended=False)

    m = np.NAN
    if np.isfinite(_wer):
        m = .5*(_stoi + (1 - _wer))
    
    return [_stoi, _wer, m]


if __name__ == "__main__":
    main(ENHANCED_FOLDER, CLEANED_FOLDER)