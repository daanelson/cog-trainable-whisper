import argparse
import glob
from datasets import Dataset, Audio, Value

"""
Example script parsing small subset of librispeech data into appropriate format for fine-tuning
"""

def process_data(root_dir, out_dir):
    audio_data = [val for val in glob.glob(f'{root_dir}/*.flac')]
    audio_data.sort()


    transcriptions = open(f"{root_dir}/174-168635.trans.txt", 'r').readlines()
    clean_text = [val.strip().split(" ", maxsplit=1) for val in transcriptions]
    text_labels = [val[1] for val in clean_text]


    if len(audio_data) != len(text_labels):
        raise Exception("Data size mismatch")

    audio_dataset = Dataset.from_dict({"audio": audio_data,
                    "sentence": text_labels})

    audio_dataset = audio_dataset.cast_column("audio", Audio(sampling_rate=16_000))
    audio_dataset = audio_dataset.cast_column("sentence", Value("string"))
    audio_dataset.save_to_disk(out_dir)
    print("Dataset parsed")

if __name__ == '__main__':
    process_data("example-data", "data-out")