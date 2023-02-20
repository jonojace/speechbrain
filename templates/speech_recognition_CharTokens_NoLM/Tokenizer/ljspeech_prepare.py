"""
Downloads and creates manifest files for speech recognition with Mini LibriSpeech.

Authors:
 * Peter Plantinga, 2021
 * Mirco Ravanelli, 2021
"""
import math
import os
from os.path import exists
import json
import logging
from speechbrain.utils.data_utils import get_all_files, download_file
from speechbrain.dataio.dataio import read_audio
from text_normalisation import cleaners
import tarfile
from tqdm import tqdm
import shlex
import subprocess
import re

logger = logging.getLogger(__name__)
LJSPEECH_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
SAMPLERATE = 22050
LJSPEECH_EXPECTED_N = 13100


def prepare_ljspeech(
        data_folder, uttids_to_excl, valid_percent, test_percent,
        save_json_train, save_json_valid, save_json_test,
        text_cleaners
):
    """
    Prepares the json files for the LJSpeech dataset.

    Downloads the dataset if its not found in the `data_folder`.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the Mini Librispeech dataset is stored.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.

    Example
    -------
    >>> data_folder = '/path/to/mini_librispeech'
    >>> prepare_mini_librispeech(data_folder, 'train.json', 'valid.json', 'test.json')
    """

    # Check if this phase is already done (if so, skip it)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return

    # If the dataset doesn't exist yet, download it
    ljspeech_folder = os.path.join(data_folder, "LJSpeech-1.1")
    if not check_folders(ljspeech_folder):
        download_ljspeech(data_folder)

    # List files and create manifest from list
    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )
    extension = [".wav"]

    wav_list = get_all_files(ljspeech_folder, match_and=extension)
    n = len(wav_list)
    assert n == LJSPEECH_EXPECTED_N

    # downsample files to 16khz if not done already
    wavs_16khz_folder = os.path.join(ljspeech_folder, 'wavs_16khz')
    if not check_folders(wavs_16khz_folder):
        def convert16k(inputfile, outputfile16k):
            command = (f'sox -c 1 -b 16 {inputfile} -t wav {outputfile16k} rate 16k')
            subprocess.call(shlex.split(command))
        print(f"Downsampling wavs to 16khz...")
        os.mkdir(wavs_16khz_folder)
        for wav_p in tqdm(wav_list):
            wavfile = wav_p.split(os.path.sep)[-1]
            outputfile16k = os.path.join(wavs_16khz_folder, wavfile)
            # print(wav_p, outputfile16k)
            convert16k(wav_p, outputfile16k)

    # filter out uttids since we do not want to train the ASR model on ids that the respeller will be trained on
    def get_uttid(wav_p):
        path_parts = wav_p.split(os.path.sep)
        uttid, _ = os.path.splitext(path_parts[-1])
        return uttid

    with open(uttids_to_excl, 'r') as f:
        lines = f.readlines()
        uttids_to_excl = [line.strip() for line in lines]
    wav_list = [wav_p for wav_p in wav_list if get_uttid(wav_p) not in uttids_to_excl]
    filtered_n = len(wav_list)
    valid_n = math.floor(valid_percent * filtered_n)
    test_n = math.floor(test_percent * filtered_n)
    train_n = filtered_n - (valid_n + test_n)
    assert train_n + valid_n + test_n == filtered_n, f"{train_n=} {valid_n=} {test_n=} {train_n+valid_n+test_n=} {filtered_n=}"

    # List of wav audio files
    wav_list_train = wav_list[:train_n]
    wav_list_valid = wav_list[train_n:train_n + valid_n]
    wav_list_test = wav_list[train_n + valid_n:]
    assert len(wav_list_train) + len(wav_list_valid) + len(wav_list_test) == filtered_n

    trans_dict = create_trans_dict(os.path.join(data_folder, 'LJSpeech-1.1', 'metadata.csv'),
                                   text_cleaners=text_cleaners)

    # Create the json files
    create_json(wav_list_train, trans_dict, save_json_train)
    create_json(wav_list_valid, trans_dict, save_json_valid)
    create_json(wav_list_test, trans_dict, save_json_test)


def create_trans_dict(ljspeech_metadata_csv_path, text_cleaners):
    def clean_text(string):
        for name in text_cleaners:
            cleaner = getattr(cleaners, name)
            if not cleaner:
                raise Exception('Unknown cleaner: %s' % name)
            string = cleaner(string)
        return string

    with open(ljspeech_metadata_csv_path, 'r') as f:
        lines = f.readlines()
    assert len(lines) == LJSPEECH_EXPECTED_N

    trans_dict = {}
    for line in lines:
        uttid, raw_text, norm_text = line.split('|')
        text = clean_text(norm_text)
        trans_dict[uttid] = text

    logger.info("Transcription files read!")
    return trans_dict


def create_json(wav_list, trans_dict, json_file):
    """
    Creates the json file given a list of wav files and their transcriptions.

    Arguments
    ---------
    wav_list : list of str
        The list of wav files.
    trans_dict : dict
        Dictionary of sentence ids and word transcriptions.
    json_file : str
        The path of the output json file
    """
    # Processing all the wav files in the list
    json_dict = {}
    for wav_file in wav_list:
        # Reading the signal (to retrieve duration in seconds)
        signal = read_audio(wav_file)
        duration = signal.shape[0] / SAMPLERATE

        # Manipulate path to get relative path and uttid
        path_parts = wav_file.split(os.path.sep)
        uttid, _ = os.path.splitext(path_parts[-1])
        relative_path = os.path.join("{data_root}", *path_parts[-3:])

        # Replace original wavs folder with downsampled one
        relative_path = re.sub(r'/wavs/', r'/wavs_16khz/', relative_path)

        # Create entry for this utterance
        json_dict[uttid] = {
            "wav": relative_path,
            "length": duration,
            "words": trans_dict[uttid],
        }

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


def extract_bz2(filename, path="."):
    with tarfile.open(filename, "r:bz2") as tar:
        print(f"Extracting {filename} using tarfile, could take some time!!! Please wait...")
        tar.extractall(path)


def download_ljspeech(destination):
    """Download dataset and unpack it.

    Arguments
    ---------
    destination : str
        Place to put dataset.
    """
    archive = os.path.join(destination, "LJSpeech-1.1.tar.bz2")
    if not exists(archive):
        download_file(LJSPEECH_URL, archive)
    extract_bz2(archive, destination)
