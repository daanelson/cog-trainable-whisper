import os
import shutil
import zipfile
from transformers import pipeline, WhisperProcessor
import torch

from cog import BasePredictor, Path, Input
from typing import Optional



class Predictor(BasePredictor):

    # NB: change from the old version: weights now refers to the fine-tuned adaptor weights, and not the underlying model weights
    def setup(self, weights: Optional[Path] = None):
        weights = "/src/training_output.zip"
        if weights is not None:
            weights = str(weights)
            if '.zip' in weights:
                out = '/src/weights_dir'
                if os.path.exists(out):
                    shutil.rmtree(out)
                with zipfile.ZipFile(weights, 'r') as zip_ref:
                    zip_ref.extractall(out)
                    weights = os.path.join(out, 'model')
                    processor = WhisperProcessor.from_pretrained(os.path.join(out, 'processor'))
        else:
            weights = "openai/whisper-small"
        self.model = pipeline(
            task="automatic-speech-recognition",
            model=weights, 
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=torch.device("cuda:0"),
            chunk_length_s=30, 
            generate_kwargs={"num_beams": 5}
        )


    def predict(
        self,
        audio_file: Path = Input(description = "audio to transcribe")
    ) -> str:

        result = self.model(str(audio_file), return_timestamps=True)
        return result["text"]
