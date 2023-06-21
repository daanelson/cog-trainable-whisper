import os
import subprocess
import time
from collections import OrderedDict
from typing import Optional

import torch
from cog import BasePredictor, Input, Path, ConcatenateIterator
from peft import PeftModel
from tensorizer import TensorDeserializer
from tensorizer.utils import no_init_or_tensor
from transformers import (
    AutoConfig,
    StoppingCriteria,
    StoppingCriteriaList, AutoModelForCausalLM,
)
from subclass import YieldingCausalLM

from config import load_tokenizer, load_model, format_prompt, maybe_download 

CACHE_DIR = "pretrained_weights"

class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class Predictor(BasePredictor):

    # NB: change from the old version: weights now refers to the fine-tuned adaptor weights, and not the underlying model weights
    def setup(self, weights: Optional[Path] = None):
        self.model = load_model(plaid_mode=True, cls=YieldingCausalLM)

        self.tokenizer = load_tokenizer()

        if weights is not None:
            self.model = self.load_fine_tuned(self.model, weights=weights)

        self.stop = StopOnTokens()

    def load_fine_tuned(self, model, weights):
        """Load fine-tuned adaptor weights using PEFT."""
        st = time.time()
        print(f"loading fine-tuned weights from {weights}")

        weights = str(weights)
        if weights.startswith("https:") or weights.startswith("gs:"):
            local_path = '/src/output_weights.zip'
            maybe_download(weights, local_path)
            weights = local_path

        if weights.endswith(".zip"):
            import zipfile
            weights = zipfile.ZipFile(weights, "r")
            # munge filename four output directory
            output_dir = os.path.basename(weights.filename).replace(".zip", "")
            weights.extractall(f"{CACHE_DIR}/{output_dir}")
            weights = f"./{CACHE_DIR}/{output_dir}"

        model = PeftModel.from_pretrained(model, weights, cache_dir=CACHE_DIR).to("cuda:0")
        print(f"weights loaded in {time.time() - st}")
        return model

    def predict(
        self,
        prompt: str = Input(
            description=f"Input Prompt.", default="What's your mood today?"
        ),
        max_tokens: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=100,
        ),
        top_p: float = Input(
            description="Valid if you choose top_p decoding. When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.01,
            le=1.0,
            default=1.0,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.01,
            le=5,
            default=0.75,
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            ge=0.01,
            le=5,
            default=1.2,
        ),
    ) -> ConcatenateIterator[str]:

        prompt_text = format_prompt(prompt)
        print(f"prompt: {prompt_text}")

        input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(
            "cuda:0"
        )
        with torch.inference_mode():
            first_token_yielded = False
            prev_ids = []
            for output in self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                num_return_sequences=1,
                num_beams=1,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stopping_criteria=StoppingCriteriaList([self.stop]),
            ):
                cur_id = output.item()
                # in order to properly handle spaces, we need to do our own tokenizing. Fun!
                # we're building up a buffer of sub-word / punctuation tokens until we hit a space, and then yielding whole words + punctuation.
                cur_token = self.tokenizer.convert_ids_to_tokens(cur_id)
        
                # skip initial newline, which this almost always yields. hack - newline id = 187.
                if not first_token_yielded and not prev_ids and cur_id == 187:
                    continue
        
                # Space is represented as "Ġ".
                # Yield previous IDs if we hit a space
                # or if the current token includes a space
                # at its start (e.g. ' is' -> 'Ġis')
                if cur_token.startswith("Ġ"):
                    if prev_ids:
                        yield self.tokenizer.decode(prev_ids)
                        prev_ids = []
        
                    prev_ids = [cur_id]
                    continue
        
                # End token
                if cur_token == "<|endoftext|>":
                    break
        
                prev_ids.append(cur_id)
        
            if prev_ids:
                yield self.tokenizer.decode(prev_ids)
                prev_ids = []
