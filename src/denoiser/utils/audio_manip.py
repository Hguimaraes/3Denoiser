import torch
import speechbrain as sb

"""
Read predictor and target for L3DAS22
"""
def read_audio(wav_files, is_test:bool=False):
    # Read the waveform
    predictor = sb.dataio.dataio.read_audio_multichannel(
        wav_files['predictors']
    )

    target = None
    if not is_test:
        target = sb.dataio.dataio.read_audio(wav_files['wave_target'])
        target = torch.unsqueeze(target, 0)

    return predictor.transpose(0, 1), target


"""
Get a random subsample of the original audio
"""
def sample(
    predictor: torch.Tensor, 
    target: torch.Tensor, 
    max_size: int
):
    samples = predictor.shape[0]
    if samples > max_size:
        offset = torch.randint(low=0, high=max_size-1, size=(1,))
        target = target[offset:(offset+max_size), :]
        predictor = predictor[offset:(offset+max_size), :]
    
    return predictor, target
