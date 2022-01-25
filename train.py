import os
import logging
import argparse
import pandas as pd
from typing import Dict
from seq2seq_model import Seq2SeqModel

logger = logging.getLogger(__name__)


class NoTestDevFound(Exception):
    pass


log_levels = {
    "i": "INFO",
    "d": "DEBUG",
    "e": "ERROR",
}

preset_model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 50,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    "evaluate_during_training": True,
    "evaluate_generated_text": True,
    "evaluate_during_training_verbose": True,
    "use_multiprocessing": False,
    "max_length": 25,
    "save_steps": 11898,
    "gradient_accumulation_steps": 1,
}


parser = argparse.ArgumentParser(
    description="This script trains an encoder decoder model for template based NER"
)
parser.add_argument(
    "input_dir",
    help="Data directory path containing train, dev, test csv files.",
)
parser.add_argument(
    "-o",
    "--output_dir",
    default="./outputs",
    help="Output directory for model and evaluation results.",
)
parser.add_argument(
    "-n",
    "--encoder_decoder_name",
    default="hf-internal-testing/tiny-random-mbart",
    help="Model checkpoint from Hugging Face.",
)
parser.add_argument(
    "-t",
    "--encoder_decoder_type",
    default="mbart",
    help="Type of the HF model to train. Available options are 'mbart' and 'bart'",
)
parser.add_argument("-s", "--manual_seed", type=int, default=7)
parser.add_argument("-e", "--num_train_epochs", type=int, default=3)
parser.add_argument("-b", "--train_batch_size", type=int, default=32)
parser.add_argument("-l", "--learning_rate", type=float, default=1e-4)
parser.add_argument("-w", "--wandb_project", type=str, default=None)
parser.add_argument("-g", "--gradient_checkpointing", action="store_true")
parser.add_argument("-p", "--sep", default=";")
parser.add_argument("-v", "--log_level", default="i")
parser.add_argument("--max_train_size", type=int, default=None)
parser.add_argument("--max_eval_size", type=int, default=None)


def read_data(
    input_dir: str, sep: str, max_train_size: int, max_eval_size: int
) -> Dict["str", pd.DataFrame]:
    datasets = {"train": None, "dev": None, "test": None}

    for filename in datasets.keys():
        filepath = os.path.join(input_dir, filename + ".csv")
        if os.path.isfile(filepath):
            logger.info(f"Loading {filename} from {filepath}.")
            data = pd.read_csv(filepath, sep=sep, header=None).values.tolist()
            data = pd.DataFrame(data, columns=["input_text", "target_text"])
            datasets[filename] = data
            logger.info(f"Shape: {data.shape}")

    if max_train_size:
        logging.info(f"Reducing train size to {max_train_size}")
        datasets["train"] = datasets["train"][:max_train_size]
    if max_eval_size:
        logging.info(f"Reducing eval size to {max_eval_size}")
        if datasets["dev"] is not None:
            datasets["dev"] = datasets["dev"][:max_eval_size]
        if datasets["test"] is not None:
            datasets["test"] = datasets["test"][:max_eval_size]

    if datasets["dev"] is None and datasets["test"] is None:
        raise NoTestDevFound(
            "Test or dev data not found. Add a dev or test file to your input dir."
        )

    return datasets


def train(args, datasets: Dict["str", pd.DataFrame]):

    encoder_decoder_name = args["encoder_decoder_name"]
    encoder_decoder_type = args["encoder_decoder_type"]
    del args["encoder_decoder_name"]
    del args["encoder_decoder_type"]

    args.update(preset_model_args)
    model = Seq2SeqModel(
        encoder_decoder_name=encoder_decoder_name,
        encoder_decoder_type=encoder_decoder_type,
        args=args,
    )

    if datasets["dev"] is not None and datasets["test"] is not None:
        model.train_model(datasets["train"], eval_data=datasets["dev"])
        results = model.eval_model(datasets["test"])
    elif datasets["test"] is not None and datasets["dev"] is None:
        model.train_model(datasets["train"], eval_data=datasets["test"])
        results = model.eval_model(datasets["test"])
    elif datasets["dev"] is not None and datasets["test"] is None:
        model.train_model(datasets["train"], eval_data=datasets["dev"])
        results = model.eval_model(datasets["dev"])

    return model, results


def main(args):
    args = vars(args)

    # Set log level
    logging.basicConfig(level=log_levels[args["log_level"]])
    del args["log_level"]

    # Read datasets
    datasets = read_data(
        input_dir=args["input_dir"],
        sep=args["sep"],
        max_train_size=args["max_train_size"],
        max_eval_size=args["max_eval_size"],
    )
    del args["input_dir"]
    del args["sep"]
    del args["max_train_size"]
    del args["max_eval_size"]
    # Start training
    model, results = train(args=args, datasets=datasets)
    print(results)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
