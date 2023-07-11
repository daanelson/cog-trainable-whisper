import os
import shutil
import zipfile
from typing import Optional

import torch
from cog import BasePredictor, Input, Path
from transformers import WhisperProcessor, pipeline


class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        if weights is not None:
            weights_url = str(weights)
            local_path = "/src/weights.zip"
            os.system(f"wget {weights_url} -O {local_path}")
            out = "/src/weights_dir"
            if os.path.exists(out):
                shutil.rmtree(out)
            with zipfile.ZipFile(local_path, "r") as zip_ref:
                zip_ref.extractall(out)
                weights = os.path.join(out, "model")
                processor = WhisperProcessor.from_pretrained(
                    os.path.join(out, "processor")
                )
        else:
            weights = "openai/whisper-small"
        self.model = pipeline(
            task="automatic-speech-recognition",
            model=weights,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=torch.device("cuda:0"),
            chunk_length_s=30,
            generate_kwargs={"num_beams": 5},
        )

    def predict(
        self, audio_file: Path = Input(description="audio to transcribe")
    ) -> str:
        result = self.model(str(audio_file), return_timestamps=True)
        return result["text"]
