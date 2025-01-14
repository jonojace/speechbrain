#!/usr/bin/env/python3
"""
Jason Fong
- Adapted the standard ASR template to recreate BiLSTM architecture trained with CTC in
  "Towards End-to-End Speech Recognition with Recurrent Neural Networks" (Graves, 2014)

Recipe for training a sequence-to-sequence ASR system with mini-librispeech.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with beam search coupled with a neural
language model.

To run this recipe, do the following:
> python train.py train.yaml

With the default hyperparameters, the system employs an LSTM encoder.
The decoder is based on a standard  GRU. Beam search coupled with an RNN language
model is used on the top of decoder probabilities.

The neural network is trained on both CTC and negative-log likelihood
targets and sub-word units estimated with Byte Pairwise Encoding (BPE)
are used as basic recognition tokens. Training is performed on the mini-librispeech
dataset. Note that this is a tiny dataset used here just to
provide a working example. To achieve a better performance you have to train with
larger datasets, such as the full LibriSpeech one. In this case, to allow the
model to converge, we pre-train it with a bigger one (trained on the full librispeech
with the seq2seq 1k BPE recipe).

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE).

This recipe assumes that the tokenizer and the LM are already trained.
To avoid token mismatches, the tokenizer used for the acoustic model is
the same use for the LM.  The recipe downloads the pre-trained tokenizer
and LM.

If you would like to train a full system from scratch do the following:
1- Train a tokenizer (see ../Tokenizer)
2- Train a language model (see ../LM)
3- Train the speech recognizer (with this code).


Authors
 * Mirco Ravanelli 2020
 * Ju-Chieh Chou 2020
 * Abdel Heba 2020
 * Peter Plantinga 2020
 * Samuele Cornell 2020
"""

import sys
import torch
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
# from mini_librispeech_prepare import prepare_mini_librispeech
from speechbrain.utils.distributed import run_on_main
import os
import wandb
try:
    from templates.speech_recognition_CharTokens_NoLM.ASR.ljspeech_prepare import prepare_ljspeech, get_uttid, append_str_to_filename
except ModuleNotFoundError:
    from ljspeech_prepare import prepare_ljspeech, get_uttid, append_str_to_filename

logger = logging.getLogger(__name__)


def dump_feats_to_dir(feats, wav_paths, dump_feats_dir, file_ext=".pt"):
    if not os.path.exists(dump_feats_dir):
        os.makedirs(dump_feats_dir)

    for i, wav_path in enumerate(wav_paths):
        dumped_feats_path = os.path.join(dump_feats_dir, get_uttid(wav_path) + file_ext)
        torch.save(feats[i, :, :], dumped_feats_path)


# Brain class for speech recognition training
class ASR(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""

    def compute_forward(self, batch, stage):
        """Runs all the computation of the CTC + seq2seq ASR. It returns the
        posterior probabilities of the CTC and seq2seq networks.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : dict
            At training time it returns predicted seq2seq log probabilities.
            If needed it also returns the ctc output log probabilities.
            At validation/test time, it returns the predicted tokens as well.
        """
        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)

        # NOTE CAREFUL!!! self.feat_lens are not mel lens but ratios from 0.0 to 1.0 (ratio of wav len to max wav len)
        feats, self.feat_lens = self.prepare_features(stage, batch.sig, batch.wav_path)
        # tokens_bos, _ = self.prepare_tokens(stage, batch.tokens_bos)

        # Running the encoder (prevent propagation to feature extraction)
        encoded_signal = self.modules.encoder(feats.detach())

        # # Embed tokens and pass tokens & encoded signal to decoder
        # embedded_tokens = self.modules.embedding(tokens_bos)
        # decoder_outputs, _ = self.modules.decoder(
        #     embedded_tokens, encoded_signal, self.feat_lens
        # )

        # # Output layer for seq2seq log-probabilities
        # logits = self.modules.seq_lin(decoder_outputs)
        # predictions = {"seq_logprobs": self.hparams.log_softmax(logits)}

        predictions = {}

        if self.is_ctc_active(stage):
            # Output layer for ctc log-probabilities
            ctc_logits = self.modules.ctc_lin(encoded_signal)
            predictions["ctc_logprobs"] = self.hparams.log_softmax(ctc_logits)
        # elif stage == sb.Stage.VALID:
        #     predictions["tokens"], _ = self.hparams.valid_search(
        #         encoded_signal, self.feat_lens
        #     )
        # elif stage == sb.Stage.TEST:
        #     predictions["tokens"], _ = self.hparams.test_search(
        #         encoded_signal, self.feat_lens
        #     )

        return predictions

    def is_ctc_active(self, stage):
        """Check if CTC is currently active.

        Arguments
        ---------
        stage : sb.Stage
            Currently executing stage.
        """
        # if stage != sb.Stage.TRAIN:
        #     return False
        # current_epoch = self.hparams.epoch_counter.current
        # return current_epoch <= self.hparams.number_of_ctc_epochs
        return True

    def prepare_features(self, stage, wavs, wav_paths):
        """Prepare features for computation on-the-fly

        Arguments
        ---------
        stage : sb.Stage
            Currently executing stage.
        wavs : tuple
            The input signals (tensor) and their lengths (tensor).
        """
        wavs, wav_lens = wavs

        dump_feats = False
        if hasattr(self.hparams, "dump_feats"):
            if self.hparams.dump_feats:
                dump_feats = True

        # Add augmentation if specified. In this version of augmentation, we
        # concatenate the original and the augment batches in a single bigger
        # batch. This is more memory-demanding, but helps to improve the
        # performance. Change it if you run OOM.
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])

            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # Feature computation and normalization
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)

        if dump_feats:
            dump_feats_to_dir(feats, wav_paths, self.hparams.dump_feats_dir)

        return feats, wav_lens

    def prepare_tokens(self, stage, tokens):
        """Double the tokens batch if features are doubled.

        Arguments
        ---------
        stage : sb.Stage
            Currently executing stage.
        tokens : tuple
            The tokens (tensor) and their lengths (tensor).
        """
        tokens, token_lens = tokens
        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens = torch.cat([tokens, tokens], dim=0)
            token_lens = torch.cat([token_lens, token_lens], dim=0)
        return tokens, token_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs. We here
        do multi-task learning and the loss is a weighted sum of the ctc + seq2seq
        costs.

        Arguments
        ---------
        predictions : dict
            The output dict from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        # Compute sequence loss against targets with EOS
        # tokens_eos, tokens_eos_lens = self.prepare_tokens(
        #     stage, batch.tokens_eos
        # )

        # loss = sb.nnet.losses.nll_loss(
        #     log_probabilities=predictions["seq_logprobs"],
        #     targets=tokens_eos,
        #     length=tokens_eos_lens,
        #     label_smoothing=self.hparams.label_smoothing,
        # )

        # Add ctc loss if necessary. The total cost is a weighted sum of
        # ctc loss + seq2seq loss
        # if self.is_ctc_active(stage):

        # Load tokens without EOS as CTC targets
        tokens, tokens_lens = self.prepare_tokens(stage, batch.tokens)
        loss = self.hparams.ctc_cost(
            predictions["ctc_logprobs"], tokens, self.feat_lens, tokens_lens
        )
        # wandb.log({'train.loss': loss})
        self.hparams.train_logger.log_dict({'iter_loss': loss})

        # loss *= 1 - self.hparams.ctc_weight
        # loss += self.hparams.ctc_weight * loss_ctc

        if stage != sb.Stage.TRAIN:
            # print("DEBUG stage", stage)

            # TODO replace with S2SBeamSearcher? but set LM weight to 0 so it uses CTC prefix scorer to do beam search
            # TODO and find alternative hypotheses
            predicted_ids = sb.decoders.ctc_greedy_decode(
                predictions["ctc_logprobs"], self.feat_lens, blank_id=self.hparams.blank_index
            )
            predicted_words = [
                self.hparams.tokenizer.decode_ids(ids).split(" ")
                for ids in predicted_ids
            ]
            target_words = [words.split(" ") for words in batch.words]

            # Monitor word error rate and character error rated at
            # valid and test time.
            self.wer_metric.append(batch.id, predicted_words, target_words)
            self.cer_metric.append(batch.id, predicted_words, target_words)

            # print(f"DEBUG {[self.hparams.tokenizer.decode_ids(ids) for ids in predicted_ids]=}")
            # print(f"DEBUG jason {tokens=}")
            # print(f"DEBUG jason {predictions['ctc_logprobs'].size()=}")
            # print(f'DEBUG jason {self.hparams.tokenizer.get_piece_size()=}')
            # for idx in range(len(self.hparams.tokenizer)):
            #     print(f'DEBUG Sentpiece {idx=} {self.hparams.tokenizer.decode_ids([idx])=}')
            # print(f'DEBUG valid predictions: {predicted_ids=}')
            # print(f'DEBUG valid predictions: {predicted_words=}')
            # print(f'DEBUG valid predictions: {target_words=}')

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # Set up statistics trackers for this stage
        # In this case, we would like to keep track of the word error rate (wer)
        # and the character error rate (cer)
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Summarize the statistics from the stage for record-keeping.
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            # print(f"DEBUG LOGGING VALID METRICS, {stage=}, {stage_stats=}")

            # Update learning rate
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats[self.hparams["metric_to_optimize"]])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={self.hparams["metric_to_optimize"]: stage_stats[self.hparams["metric_to_optimize"]]}, min_keys=[self.hparams["metric_to_optimize"]],
            )

        # We also write statistics about test data to stdout and to the logfile.
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.


    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Dictionary containing "train", "valid", and "test" keys that correspond
        to the DynamicItemDataset objects.
    """
    def remove_whitespace(s):
        return s.replace(" ", "").replace("|", "")

    # Define audio pipeline. In this case, we simply read the path contained
    # in the variable wav with the audio reader.
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig", "wav_path", "utt_id")
    def audio_pipeline(wav_path):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav_path)
        yield sig

        yield wav_path

        utt_id = wav_path.split("/")[-1].split(".")[0]
        yield utt_id

    # Define text processing pipeline. We start from the raw text and then
    # encode it using the tokenizer. The tokens with BOS are used for feeding
    # decoder during training, the tokens with EOS for computing the cost function.
    # The tokens without BOS or EOS is for computing CTC loss.
    @sb.utils.data_pipeline.takes("words")
    @sb.utils.data_pipeline.provides(
        "words", "tokens_list", "tokens"
    )
    def text_pipeline(words):
        """Processes the transcriptions to generate proper labels

        NB Make sure that you yield exactly what is defined above in @sb.utils.data_pipeline.provides()"""
        if hparams["no_whitespace"] is not None and hparams["no_whitespace"]:
            # print("DEBUG1", words)
            words = remove_whitespace(words)
            # print("DEBUG2", words)
        yield words

        tokens_list = hparams["tokenizer"].encode_as_ids(words)
        assert len(tokens_list) > 0, f"Something is wrong with the tokenizer."
        yield tokens_list

        tokens = torch.LongTensor(tokens_list)
        yield tokens

    # Define datasets from json data manifest file
    # Define datasets sorted by ascending lengths for efficiency
    datasets = {}
    data_folder = hparams["data_folder"]

    dump_feats_append_str = ""
    if hparams["dump_feats"]:
        dump_feats_append_str = "FOR_DUMPING_FEATS"

    data_info = {
        "train": append_str_to_filename(hparams["train_annotation"], dump_feats_append_str),
        "valid": append_str_to_filename(hparams["valid_annotation"], dump_feats_append_str),
        "test": append_str_to_filename(hparams["test_annotation"], dump_feats_append_str),
    }

    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": data_folder},
            dynamic_items=[audio_pipeline, text_pipeline],
            output_keys=[
                "id",
                "sig",
                "wav_path",
                "utt_id",
                "words",
                "tokens",
            ],
        )
        hparams[f"{dataset}_dataloader_opts"]["shuffle"] = False

    # Sorting training data with ascending order makes the code  much
    # faster  because we minimize zero-padding. In most of the cases, this
    # does not harm the performance.
    if hparams["sorting"] == "ascending":
        datasets["train"] = datasets["train"].filtered_sorted(sort_key="length")
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        datasets["train"] = datasets["train"].filtered_sorted(
            sort_key="length", reverse=True
        )
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        hparams["train_dataloader_opts"]["shuffle"] = True
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
    return datasets

if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # init WANDB logger
    hparams["train_logger"].init(hparams)

    # print("debug hparams", hparams.__dict__)
    # wandb.config.update(hparams, allow_val_change=True)  # now we update config as yaml has been properly parsed

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # save annotation files in the data dir
    hparams["train_annotation"] = os.path.join(hparams["data_folder"], hparams["train_annotation"])
    hparams["valid_annotation"] = os.path.join(hparams["data_folder"], hparams["valid_annotation"])
    hparams["test_annotation"] = os.path.join(hparams["data_folder"], hparams["test_annotation"])

    # Data preparation, to be run on only one process.
    if hparams["corpus_name"] == 'ljspeech':
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
            "dump_feats": hparams["dump_feats"],
        }
    elif hparams["corpus_name"] == 'mini_librispeech':
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

    # We can now directly create the datasets for training, valid, and test
    datasets = dataio_prepare(hparams)

    # In this case, pre-training is essential because mini-librispeech is not
    # big enough to train an end-to-end model from scratch. With bigger dataset
    # you can train from scratch and avoid this step.
    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    # run_on_main(hparams["pretrainer"].collect_files)
    # hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.

    print(f"debug! {len(datasets['train'])=}")
    print(f"debug! {len(datasets['valid'])=}")

    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
        dump_feats=hparams["dump_feats"],
    )

    # Load best checkpoint for evaluation
    test_stats = asr_brain.evaluate(
        test_set=datasets["test"],
        min_key=hparams["metric_to_optimize"],
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
