import argparse
import glob
import json
import os
import shutil
import subprocess
import tarfile

from cog import BasePredictor, Input, Path
from datasets import Audio, Dataset, Value

"""
Example script parsing audio & transcription data into fixed dataset for whisper fine-tuning
"""


def untar(tarball, target_dir):
    """Untars a tarball. Flattens all directory structure and returns a list of files."""
    # Check if the target directory exists. If not, create it.
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    try:
        with tarfile.open(tarball) as tf:
            for member in tf.getmembers():
                if member.isdir():
                    continue

                # Ensure we remove all leading directories from file path
                path = os.path.join(target_dir, (member.name.split("/")[-1]))

                # Extract the file (note we need to use `tf.extractfile` because
                # `tf.extractall` doesn't let us control the output path per file)
                with tf.extractfile(member) as f, open(path, "wb") as out:
                    out.write(f.read())
    except Exception as e:
        print(f"An error occurred while extracting {tarball}. Error: {e}")


def make_tarfile(output_filename, source_dir):
    # Check if the source directory exists
    if not os.path.isdir(source_dir):
        print(f"Source directory {source_dir} does not exist.")
        return

    with tarfile.open(output_filename, "w:gz") as tar:
        for root, _, files in os.walk(source_dir):
            for file in files:
                # Create full path to the file
                full_path = os.path.join(root, file)
                # Add file to the tar file with a relative path
                # This will not include the directory itself
                tar.add(full_path, arcname=os.path.relpath(full_path, source_dir))
    return Path(output_filename)


def download_file(url, folder):
    """Downloads and stores files"""
    file_name = url.split("/")[-1]
    subprocess.check_call(["pget", url, os.path.join(folder, file_name)])
    return file_name


def write_file(text, folder, audio_fname):
    """Writes text file consistent with audio filename"""
    txt_fname = audio_fname.split(".", [0])
    with open(os.path.join(folder, f"{txt_fname}.txt"), "w") as f:
        f.writelines(text)


class WhisperDatasetProcessor(BasePredictor):
    def setup(self):
        self.audio_folder = "/src/audio"
        self.text_folder = "/src/text"
        self.out_path = "dataset_out"

        def _clear_if_exists(fpath):
            if os.path.exists(fpath):
                shutil.rmtree(fpath)

        _clear_if_exists(self.audio_folder)
        _clear_if_exists(self.text_folder)
        _clear_if_exists(self.out_path)

    def predict(
        self,
        audio_files: Path = Input(
            description="tarball with list of audio files", default=None
        ),
        text_files: Path = Input(
            description="tarball with list of transcriptions", default=None
        ),
    ) -> Path:
        untar(audio_files, self.audio_folder)
        untar(text_files, self.text_folder)

        self.build_dataset()
        return make_tarfile("whisper_dataset.tar.gz", self.out_path)

    def build_dataset(self):
        audio_folder = self.audio_folder
        text_folder = self.text_folder
        out_path = self.out_path
        audio_data = [
            os.path.join(audio_folder, val) for val in os.listdir(audio_folder)
        ]

        def parse_text(val):
            with open(os.path.join(text_folder, val)) as f:
                text = f.readlines()
                text = "\n".join(text)
            return text.strip()

        text_labels = [parse_text(val) for val in os.listdir(text_folder)]

        if len(audio_data) != len(text_labels):
            raise Exception(
                f"Data size mismatch, {len(audio_data)} audio files and {len(text_labels)} text files were provided"
            )

        audio_dataset = Dataset.from_dict(
            {"audio": audio_data, "sentence": text_labels}
        )

        audio_dataset = audio_dataset.cast_column("audio", Audio(sampling_rate=16_000))
        audio_dataset = audio_dataset.cast_column("sentence", Value("string"))
        audio_dataset.save_to_disk(out_path)

        print("Dataset built")
        return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio and Text Predictor")

    parser.add_argument(
        "--audio-files",
        type=Path,
        help="Tarball with list of audio files",
        default=None,
    )

    parser.add_argument(
        "--transcript-files",
        type=Path,
        help="Tarball with list of transcriptions",
        default=None,
    )

    args = parser.parse_args()

    p = WhisperDatasetProcessor()
    p.setup()
    p.predict(audio_files=args.audio_files, text_files=args.transcript_files)
