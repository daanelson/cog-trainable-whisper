import argparse
import glob
import json
import os
import tarfile
import subprocess
from datasets import Dataset, Audio, Value
from cog import BasePredictor, Input, Path

"""
Example script parsing audio & transcription data into fixed dataset for whisper fine-tuning
"""

def process_data(root_dir, out_dir):
    audio_data = [val for val in glob.glob(f"{root_dir}/*.flac")]
    audio_data.sort()

    transcriptions = open(f"{root_dir}/174-168635.trans.txt", "r").readlines()
    clean_text = [val.strip().split(" ", maxsplit=1) for val in transcriptions]
    text_labels = [val[1] for val in clean_text]

    if len(audio_data) != len(text_labels):
        raise Exception("Data size mismatch")

    audio_dataset = Dataset.from_dict({"audio": audio_data, "sentence": text_labels})

    audio_dataset = audio_dataset.cast_column("audio", Audio(sampling_rate=16_000))
    audio_dataset = audio_dataset.cast_column("sentence", Value("string"))
    audio_dataset.save_to_disk(out_dir)
    print("Dataset parsed")


def untar(tarball, target_dir):
    # Check if the target directory exists. If not, create it.
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    try:
        with tarfile.open(tarball) as tf:
            tf.extractall(path=target_dir)
    except Exception as e:
        print(f"An error occurred while extracting {tarball}. Error: {e}")

def make_tarfile(output_filename, source_dir):
    # Check if the source directory exists
    if not os.path.isdir(source_dir):
        print(f"Source directory {source_dir} does not exist.")
        return

    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
    return Path(output_filename)

def download_file(url, folder, index):
    """Renaming files s.t."""
    # extension = url.split('.')[-1]
    subprocess.check_call(["pget", url, folder])
    return url.split('/')[-1]

def write_file(text, folder, audio_fname):
    """Writes text file consistent with audio filename"""
    txt_fname = audio_fname.split(".", [0])
    with open(os.path.join(folder, f'{txt_fname}.txt'), 'w') as f:
        f.writelines(text)


class WhisperDatasetProcessor(BasePredictor):

    def setup(self):
        self.audio_folder = "/src/audio"
        self.text_folder = "/src/text"
        self.out_path = "dataset_out"
        pass

    def predict(self,
            audio_files: Path = Input(description="tarball with list of audio files", default=None),
            text_files: Path = Input(description="tarball with list of transcriptions", default=None),
            jsonl_data: Path = Input(description="jsonl file with list of {'audio':<audio_url>', 'sentence':<transcription>})", default=None)
        ) -> Path:
        if audio_files and text_files:
            untar(audio_files, self.audio_folder)
            untar(text_files, self.ext_folder)
        elif jsonl_data:
            self.parse_jsonl(jsonl_data)
        else:
            raise ValueError("You need to pass either audio & text or a jsonl of files")
        
        self.build_dataset()
        return make_tarfile('whisper_dataset.tar.gz', self.out_path)
        
    def parse_jsonl(self, jsonl_path):
        data = []
        with open(jsonl_path, "r") as f:
            for line in f:
                json_object = json.loads(line)
                data.append(json_object)
        for ind, row in enumerate(data):
            audio_fname = download_file(row['audio'], self.audio_folder, ind)
            if 'https:' in row['sentence']:
                download_file(row['sentence'], self.text_folder, ind)
            else:
                write_file(row['sentence'], self.text_folder, audio_fname)

    def build_dataset(self):
        audio_folder = self.audio_folder
        text_folder = self.text_folder
        out_path = self.out_path
        audio_data = [val for val in os.listdir(audio_folder)]

        text_labels = [open(val).strip().split(" ", maxsplit=1) for val in os.listdir(text_folder)]

        if len(audio_data) != len(text_labels):
            raise Exception(f"Data size mismatch, {len(audio_data)} audio files and {len(text_labels)} text files were provided")

        audio_dataset = Dataset.from_dict({"audio": audio_data, "sentence": text_labels})

        audio_dataset = audio_dataset.cast_column("audio", Audio(sampling_rate=16_000))
        audio_dataset = audio_dataset.cast_column("sentence", Value("string"))
        audio_dataset.save_to_disk(out_path)

        print("Dataset built")
        return out_path
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio and Text Predictor')

    parser.add_argument('--audio_files', 
                        type=Path, 
                        help='Tarball with list of audio files', 
                        default=None)
    
    parser.add_argument('--text_files', 
                        type=Path, 
                        help='Tarball with list of transcriptions', 
                        default=None)

    parser.add_argument('--jsonl', 
                        type=Path, 
                        help="JSONL file with list of {'audio':<audio_url>, 'sentence':<transcription>}", 
                        default=None)

    args = parser.parse_args()
    
    p = WhisperDatasetProcessor()
    p.setup()
    p.predict(audio_files=args.audio_files, text_files=args.text_files, jsonl=args.jsonl)

