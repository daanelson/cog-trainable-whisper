"""
Cog implementation of a trainable whisper model. 
Lovingly borrowed from https://huggingface.co/blog/fine-tune-whisper
"""
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, HfArgumentParser
from datasets import load_dataset, DatasetDict, load_from_disk
import multiprocessing
import torch
import evaluate

from dataclasses import dataclass, field
from typing import Any, Dict, List, Union, Optional


@dataclass
class WhisperTrainingArguments():
    model_name: str
    whisper_language: Optional[str]
    train_data: str
    eval_data: Optional[str]
    local_output_dir: str


def train(whisper_args: WhisperTrainingArguments, seq2seq_args: Seq2SeqTrainingArguments):
    model_name = f"openai/whisper-{whisper_args.model_name}"
    processor = WhisperProcessor.from_pretrained(model_name, language=whisper_args.whisper_language, task="transcribe")
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer

    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids 
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch
    
    cpu_count = multiprocessing.cpu_count()

    print("Loading train dataset...")
    train_dataset = load_from_disk(whisper_args.train_data)
    train_dataset = train_dataset.map(prepare_dataset, num_proc=cpu_count)

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
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{model_name}")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    eval_dataset = None
    if whisper_args.eval_data:
        print("Loading eval dataset...")
        eval_dataset = load_from_disk(whisper_args.eval_data)
        eval_dataset = eval_dataset.map(prepare_dataset, num_proc=cpu_count)

    trainer = Seq2SeqTrainer(
        args=seq2seq_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.model.save_pretrained(whisper_args.local_output_dir)

if __name__ == '__main__':

    parser = HfArgumentParser([WhisperTrainingArguments, Seq2SeqTrainingArguments])
    whisper_args, seq2seq_args = parser.parse_args_into_dataclasses()
    train(whisper_args, seq2seq_args)
