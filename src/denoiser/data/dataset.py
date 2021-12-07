import torch
import speechbrain
from speechbrain.dataio.dataset import DynamicItemDataset

from denoiser.utils import sample
from denoiser.utils import read_audio

def create_datasets(hparams):
    datasets = {}

    @speechbrain.utils.data_pipeline.takes("wav_files")
    @speechbrain.utils.data_pipeline.provides("predictor", "target")
    def audio_pipeline(wav_files):
        predictor, target = read_audio(wav_files)

        # Sampling procedure
        if hparams['train_sampling']:
            max_size = hparams['max_train_sample_size']
            predictor, target = sample(predictor, target, max_size)

        return predictor, target


    @speechbrain.utils.data_pipeline.takes("wav_files")
    @speechbrain.utils.data_pipeline.provides("predictor", "target")
    def audio_pipeline_valid(wav_files):
        predictor, target = read_audio(wav_files)
        return predictor, target


    @speechbrain.utils.data_pipeline.takes("wav_files")
    @speechbrain.utils.data_pipeline.provides("predictor")
    def audio_pipeline_test(wav_files):
        predictor, _ = read_audio(wav_files, is_test=True)
        return predictor
    
    dynamic_items_map = {
        'train': [audio_pipeline],
        'valid': [audio_pipeline_valid],
        'test': [audio_pipeline_test]
    }

    for set_ in ['train', 'valid', 'test']:
        output_keys = ["id", "predictor", "length"]
        dynamic_items = dynamic_items_map[set_]

        if set_ != 'test':
            output_keys.extend(["target", "transcript"])
        
        # Construct the dynamic item dataset
        datasets[set_] = DynamicItemDataset.from_json(
            json_path=hparams[f"{set_}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=dynamic_items,
            output_keys=output_keys,
        ).filtered_sorted(sort_key="length", reverse=False)

    return datasets