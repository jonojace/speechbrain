#!/usr/bin/env/python3
"""Script for training a BPE tokenizer on the top of CSV or JSON annotation files.
The tokenizer converts words into sub-word units that can be used to train a
language (LM) or an acoustic model (AM).
When doing a speech recognition experiment you have to make
sure that the acoustic and language models are trained with
the same tokenizer. Otherwise, a token mismatch is introduced
and beamsearch will produce bad results when combining AM and LM.

To run this recipe, do the following:
> python train.py tokenizer.yaml


Authors
 * Abdel Heba 2021
 * Mirco Ravanelli 2021
"""

import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from mini_librispeech_prepare import prepare_mini_librispeech
from ljspeech_prepare import prepare_ljspeech

if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation, to be run on only one process.
    if hparams["corpus_name"] == 'ljspeech':
        print("Prepping ljspeech")
        dataprep_fn = prepare_ljspeech
        dataprep_kwargs = {
            "data_folder": hparams["data_folder"],
            "uttids_to_excl": hparams["uttids_to_excl"],
            "valid_percent": hparams["valid_percent"],
            "test_percent": hparams["test_percent"],
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "save_json_test": hparams["test_annotation"],
            "text_cleaners": hparams["text_cleaners"],
        }
    elif hparams["corpus_name"] == 'mini_librispeech':
        print("Prepping mini librispeech")
        dataprep_fn = prepare_mini_librispeech
        dataprep_kwargs = {
            "data_folder": hparams["data_folder"],
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "save_json_test": hparams["test_annotation"],
        }
    else:
        raise ValueError("unknown corpus")

    sb.utils.distributed.run_on_main(dataprep_fn, kwargs=dataprep_kwargs)

    # Train tokenizer
    hparams["tokenizer"]()
