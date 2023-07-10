import os
import shutil
import tarfile
import zipfile
from subprocess import run

import torch
from cog import BaseModel, Input, Path

MODEL_OUT = "/src/model_out"
CHECKPOINT_DIR = "checkpoints"
SAVE_STRATEGY = "epoch"
TRAIN_PATH = "/src/train_data"
EVAL_PATH = "/src/eval_data"


class TrainingOutput(BaseModel):
    weights: Path


def extract_dataset(tarball, target_dir):
    # Check if the target directory exists. If not, create it.
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    try:
        with tarfile.open(tarball) as tf:
            tf.extractall(path=target_dir)
        return target_dir
    except Exception as e:
        print(f"An error occurred while extracting {tarball}. Error: {e}")

def train(
    train_data: Path = Input(
        description="path to data file to use for fine-tuning your model"
    ),
    eval_data: Path = Input(
        description="path to optional evaluation data file to use for model eval",
        default=None,
    ),
    whisper_language: str = Input(
        description="language for fine-tuning whisper if dataset is monolingual",
        default=None,
    ),
    model_name: str = Input(
        description="which whisper model do you want to fine-tune?",
        choices=["tiny", "small", "medium", "large-v2"],
        default="small",
    ),
    per_device_train_batch_size: int = Input(
        description="batch size per GPU", default=16, ge=1
    ),
    gradient_accumulation_steps: int = Input(
        description="number of training steps to update gradient for before performing a backward pass",
        default=1,
    ),
    learning_rate: float = Input(
        description="learning rate, for learning!", default=2e-5, ge=0
    ),
    warmup_ratio: float = Input(
        description="pct of steps for a linear learning rate warmup",
        ge=0,
        le=0.5,
        default=0.03,
    ),
    num_train_epochs: int = Input(
        description="number of training epochs", ge=1, default=1
    ),
    max_steps: int = Input(
        description="number of steps to run training for, supersedes num_train_epochs",
        default=-1,
    ),
    logging_steps: int = Input(
        description="number of steps between logging epoch & loss", default=1
    ),
) -> TrainingOutput:
    # here we define other default arguments to train that we aren't exposing to users
    gradient_checkpointing = True
    fp16 = True
    predict_with_generate = True
    local_output_dir = MODEL_OUT
    output_dir = "checkpoints"

    if eval_data is not None:
        # default arguments if we have eval data
        evaluation_strategy = "epoch"
        per_device_eval_batch_size = per_device_train_batch_size

    # this is a little odd - it takes all variables defined up til this point and then passes them
    # to the training process.
    # it also avoids writing the same argument configuration in like five different places, and
    # lets us define defaults once, here. So we're doing it.
    args = locals()

    train_path = extract_dataset(train_data, TRAIN_PATH)
    args["train_data"] = train_path
    if eval_data is not None:
        eval_path = extract_dataset(eval_data, EVAL_PATH)
        args["eval_data"] = eval_path

    root_path = os.getcwd()
    deepspeed_config = os.path.join(root_path, "ds_config/ds_z3_fp16_config.json")

    # running to 1x the weights download here instead of 4x after the deepspeed kickoff
    # maybe_download()

    os.makedirs(local_output_dir, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    num_gpus_flag = f"--num_gpus={num_gpus}"

    print(f"Local Output Dir: {output_dir}")
    print(f"Number of GPUs: {num_gpus}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_DATASETS_CACHE"] = "/src/.hf-cache"

    flags = [f"--{k}={v}" for k, v in args.items() if v is not None]

    cmd = [
        "deepspeed",
        num_gpus_flag,
        "--module",
        "training.trainer",
        f"--deepspeed={deepspeed_config}",
    ] + flags

    print(f"Running training with the following command: {cmd}")

    res = run(cmd)

    if res.returncode != 0:
        raise Exception(
            f"Training failed! Process returned error code {res}. Check the logs for details."
        )

    out_path = "training_output.zip"

    directory = Path(MODEL_OUT)
    with zipfile.ZipFile(out_path, "w") as zip:
        for file_path in directory.rglob("*"):
            print(file_path)
            zip.write(file_path, arcname=file_path.relative_to(directory))

    return TrainingOutput(weights=Path(out_path))
