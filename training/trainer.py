# import argparse
# import copy
# import json
# import os

# import torch
# from cog import Input, Path
# from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_int8_training
# from torch.utils.data import Dataset
# from transformers import Trainer, TrainingArguments, AutoModelForCausalLM

# from config import DEFAULT_MODEL_NAME, load_model, load_tokenizer

###
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, DatasetDict, load_from_disk


import torch
import evaluate

from dataclasses import dataclass
from typing import Any, Dict, List, Union


# arguments
MODEL_NAME = "openai/whisper-small"
WHISPER_LANGUAGE = "Hindi"

@dataclass
class TrainingArgs():
    model_name: str
    whisper_language: str
    dataset_path: str


# processing data
def train(args: TrainingArgs):
    processor = WhisperProcessor.from_pretrained(args.model_name, language=args.whisper_language, task="transcribe")
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer

# dataset parsing - TODO (read other implementation)
## TOY IMPLEMENTATION, REPLACE THIS: 

# common_voice = DatasetDict()

# common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="train+validation")
# common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test")
# common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

# from datasets import Audio

# common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))




    dataset = load_from_disk(args.dataset_path)

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

# common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=32)

dataset = dataset.map(prepare_dataset, num_proc=4)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


metric = evaluate.load("wer")

def compute_metrics(pred):
    tokenizer = processor
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# Loading model 
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-hi",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    #generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    #report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    #push_to_hub=True,
)


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset,
    eval_dataset=dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()
trainer.save_model("a model")

# ###

# CHECKPOINT_DIR = "checkpoints"
# SAVE_STRATEGY = "epoch"
# DIST_OUT_DIR = "tmp/model"
# IGNORE_INDEX = -100

# class DatasetBuilder:
#     """Dataset agnostic class to take in input_ids and labels and spit out tokens"""

#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer

#     def batch_tokenize(self, texts):
#         """Tokenizes text. Presently doesn't pad inputs, just returns input ids."""
#         tokenized = [
#             self.tokenizer(
#                 prompt, return_tensors="pt", padding="longest", truncation=True
#             ).input_ids
#             for prompt in texts
#         ]
#         return tokenized

#     def construct_dataset(self, input_data):
#         prompts = [val["prompt"] for val in input_data]
#         tokenized_input_ids = self.batch_tokenize(prompts)
#         labels = [val["completion"] for val in input_data]
#         tokenized_labels = self.batch_tokenize(labels)
#         return TuneDataset(tokenized_input_ids, tokenized_labels)


# class CausalDatasetBuilder(DatasetBuilder):
#     """Builds generative dataset for Causal LM."""

#     def __init__(self, tokenizer, train_on_prompt=True):
#         super().__init__(tokenizer)
#         self.train_on_prompt = train_on_prompt

#     def construct_dataset(self, input_data):
#         labels = [
#             val["prompt"] + "\n" + val["completion"] + self.tokenizer.eos_token
#             for val in input_data
#         ]
#         input_ids = [val.squeeze() for val in self.batch_tokenize(labels)]
#         labels = copy.deepcopy(input_ids)
#         if self.train_on_prompt:
#             return TuneDataset(input_ids, labels)
#         # masking prompt
#         prompts = [val["prompt"] for val in input_data]
#         tokenized_prompts = self.batch_tokenize(prompts)
#         prompt_lens = [val.shape[1] for val in tokenized_prompts]

#         for label, source_len in zip(labels, prompt_lens):
#             label[:source_len] = IGNORE_INDEX
#         return TuneDataset(input_ids, labels)


# class TuneDataset(Dataset):
#     """Dead simple torch dataset wrapper. Attention masks are created in collator"""

#     def __init__(self, input_ids, labels):
#         self.input_ids = input_ids
#         self.labels = labels

#     def __len__(self):
#         return len(self.input_ids)

#     def __getitem__(self, i):
#         return dict(input_ids=self.input_ids[i], labels=self.labels[i])


# class SequenceDataCollator:
#     """Collate examples for dynamic batch construction in supervised fine-tuning."""

#     def __init__(self, tokenizer, multiple_of=None):
#         self.tokenizer = tokenizer
#         self.multiple_of = multiple_of
#         self.cache_count = 0

#     def pad_to_multiple(self, tensor, value):
#         # taking advantage of tensor cores, perhaps
#         multiple = self.multiple_of
#         target_length = (tensor.size(0) + multiple - 1) // multiple * multiple
#         return torch.nn.functional.pad(
#             tensor, (0, target_length - tensor.size(0)), value=value
#         )

#     def __call__(self, instances):
#         input_ids, labels = tuple(
#             [instance[key] for instance in instances] for key in ("input_ids", "labels")
#         )
#         if self.multiple_of:
#             input_ids = [
#                 self.pad_to_multiple(val, self.tokenizer.pad_token_id)
#                 for val in input_ids
#             ]
#             labels = [self.pad_to_multiple(val, -100) for val in labels]

#         input_ids = torch.nn.utils.rnn.pad_sequence(
#             input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
#         )
#         labels = torch.nn.utils.rnn.pad_sequence(
#             labels, batch_first=True, padding_value=-100
#         )  # -100 tells torch to ignore these tokens in loss computation.

#         # print(f"rank: {os.environ['RANK']}, cur memory: {torch.cuda.memory_allocated()}, max allocated: {torch.cuda.max_memory_allocated()}, peak memory: {torch.cuda.max_memory_reserved()}")
#         if self.cache_count < 1:
#             torch.cuda.empty_cache()
#             # print(f"rank: {os.environ['RANK']} emptying cache ")
#             self.cache_count += 1

#         return dict(
#             input_ids=input_ids,
#             labels=labels,
#             attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
#         )


# def load_data(path):
#     if path.suffix == ".json":
#         return load_json(path)
#     elif path.suffix == ".jsonl":
#         return load_jsonl(path)
#     else:
#         raise Exception(
#             f"file type {path} not supported. Currently supported types are json, jsonl"
#         )


# def load_jsonl(path):
#     data = []
#     with open(path, "r") as f:
#         for line in f:
#             json_object = json.loads(line)
#             data.append(json_object)
#     return data


# def load_json(path):
#     """Loads a single json blob"""
#     with open(path, "r") as f:
#         data = json.load(f)
#     return data


# def train(
#     train_data: Path = Input(
#         description="path to data file to use for fine-tuning your model"
#     ),
#     eval_data: Path = Input(
#         description="path to optional evaluation data file to use for model eval",
#         default=None,
#     ),
#     train_batch_size: int = Input(description="batch size per GPU", default=8, ge=1),
#     gradient_accumulation_steps: int = Input(
#         description="number of training steps to update gradient for before performing an optimizer step",
#         default=8,
#     ),
#     lr_scheduler_type: str = Input(
#         description="learning rate scheduler",
#         default="cosine",
#         choices=[
#             "linear",
#             "cosine",
#             "cosine_with_restarts",
#             "polynomial",
#             "inverse_sqrt",
#             "constant",
#             "constant_with_warmup",
#         ],
#     ),
#     learning_rate: float = Input(
#         description="learning rate, for learning!", default=2e-4, ge=0
#     ),
#     warmup_ratio: float = Input(
#         description="pct of steps for a linear learning rate warmup",
#         ge=0,
#         le=0.5,
#         default=0.03,
#     ),
#     num_train_epochs: int = Input(
#         description="number of training epochs", ge=1, default=1
#     ),
#     max_steps: int = Input(
#         description="number of steps to run training for, supersedes num_train_epochs",
#         default=-1,
#         ge=0,
#     ),
#     logging_steps: int = Input(
#         description="number of steps between logging epoch & loss", default=1
#     ),
#     local_output_dir: str = None,
#     deepspeed: str = None,
#     local_rank: int = -1,
#     lora_rank: int = 8,
#     lora_alpha: int = 16,
#     lora_dropout: float = 0.1,
#     lora_target_modules: str = 'q_proj,v_proj'
# ) -> None:
#     print("Loading model...")

#     # issue w/multi-gpu downloads in tensorizer
#     torch.cuda.set_device(int(os.environ['RANK']))
#     model = load_model(plaid_mode=False, cls=AutoModelForCausalLM)
#     tokenizer = load_tokenizer()

#     model.resize_token_embeddings(len(tokenizer))

#     peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    
#     model = get_peft_model(model, peft_config)
#     model.print_trainable_parameters()

#     print(f"Loading dataset {train_data}...")
#     print(train_data)
#     train_data = load_data(train_data)
#     p = CausalDatasetBuilder(tokenizer)
#     train_dataset = p.construct_dataset(train_data)
#     eval_dataset = None
#     if eval_data:
#         eval_data = load_json(eval_data)
#         eval_dataset = p.construct_dataset(eval_data)
#     torch.cuda.empty_cache()

#     print("Training...")
#     trainer = Trainer(
#         model=model,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         args=TrainingArguments(
#             output_dir=CHECKPOINT_DIR,
#             per_device_train_batch_size=train_batch_size,
#             gradient_accumulation_steps=gradient_accumulation_steps,
#             save_strategy="no",
#             logging_steps=logging_steps,
#             lr_scheduler_type=lr_scheduler_type,
#             warmup_ratio=warmup_ratio,
#             num_train_epochs=num_train_epochs,
#             learning_rate=learning_rate,
#             deepspeed=deepspeed,
#             max_steps=max_steps,
#             tf32=True,
#             fp16=True,
#             half_precision_backend="cuda_amp",
#             local_rank=local_rank,
#         ),
#         data_collator=SequenceDataCollator(tokenizer, 8),  # depends on bf16 value
#     )
#     trainer.train()
#     trainer.model.save_pretrained(local_output_dir)
#     return


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Fine-tune a language model on a text dataset"
#     )
#     parser.add_argument(
#         "--train_data", type=Path, required=True, help="Path to the json dataset"
#     )
#     parser.add_argument(
#         "--eval_data",
#         type=Path,
#         required=False,
#         help="Path to the json dataset",
#         default=None,
#     )
#     parser.add_argument(
#         "--num_train_epochs", type=int, help="Number of training epochs", default=1
#     )
#     parser.add_argument(
#         "--learning_rate",
#         type=float,
#         default=2e-5,
#         help="Learning rate for the optimizer",
#     )
#     parser.add_argument(
#         "--train_batch_size", type=int, default=4, help="Batch size for training"
#     )
#     parser.add_argument(
#         "--warmup_ratio",
#         type=float,
#         default=0.03,
#         help="Number of warmup steps for the learning rate scheduler",
#     )
#     parser.add_argument(
#         "--max_steps",
#         type=int,
#         default=0,
#         help="Number of training steps to run, overrides num_train_epochs, useful for testing",
#     )
#     parser.add_argument(
#         "--gradient_accumulation_steps",
#         type=int,
#         default=8,
#         help="Number of training steps to run, overrides num_train_epochs, useful for testing",
#     )
#     parser.add_argument("--logging_steps", type=int, default=1)
#     parser.add_argument(
#         "--lr_scheduler_type",
#         type=str,
#         default="cosine",
#     )
#     parser.add_argument(
#         "--deepspeed", type=str, default=None, help="Path to deepspeed config file."
#     )
#     parser.add_argument(
#         "--local_output_dir",
#         type=str,
#         help="Write directly to this local path",
#         required=True,
#     )
#     parser.add_argument(
#         "--local_rank",
#         type=int,
#         default=-1,
#         help="Provided by deepspeed to identify which instance this process is when performing multi-GPU training.",
#     )
#     parser.add_argument(
#         "--lora_rank",
#         type=int,
#         default=8,
#         help="Number of training steps to run, overrides num_train_epochs, useful for testing",
#     )
#     parser.add_argument(
#         "--lora_alpha",
#         type=int,
#         default=16,
#         help="Number of training steps to run, overrides num_train_epochs, useful for testing",
#     )
#     parser.add_argument(
#         "--lora_dropout",
#         type=float,
#         default=0.4,
#         help="Number of training steps to run, overrides num_train_epochs, useful for testing",
#     )
#     parser.add_argument(
#         "--lora_target_modules",
#         type=str, 
#         default=None,
#         help="Comma-separated list of lora modules to target, i.e. 'q_proj,v_proj'. Leave blank for default"
#     )
#     some_args = parser.parse_args()
#     train(**vars(some_args))

