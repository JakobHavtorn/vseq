import os

from types import SimpleNamespace

from vseq.settings import SOURCE_DIRECTORY


LIBRISPEECH_TRAIN = "libri_train"
LIBRISPEECH_TRAIN_CLEAN_100 = "libri_train_clean_100"
LIBRISPEECH_TRAIN_CLEAN_360 = "libri_train_clean_360"
LIBRISPEECH_TRAIN_OTHER_500 = "libri_train_other_500"
LIBRISPEECH_DEV_CLEAN = "libri_dev_clean"
LIBRISPEECH_DEV_OTHER = "libri_dev_other"
LIBRISPEECH_TEST_CLEAN = "libri_test_clean"
LIBRISPEECH_TEST_OTHER = "libri_test_other"

LIBRISPEECH_LM = "librispeech_lm"

BILLION_TRAINING = "billion_training"
BILLION_HELDOUT = "billion_heldout"

PENN_TREEBANK_TRAIN = "ptb_train"
PENN_TREEBANK_VALID = "ptb_valid"
PENN_TREEBANK_TEST = "ptb_test"

MIDI_PIANO = "midi_piano"
MIDI_PIANO_TRAIN = "midi_piano_train"
MIDI_PIANO_VALID = "midi_piano_valid"
MIDI_PIANO_TEST = "midi_piano_test"
MIDI_CHORALES_TRAIN = "midi_jsb_chorales_train"
MIDI_CHORALES_VALID = "midi_jsb_chorales_valid"
MIDI_CHORALES_TEST = "midi_jsb_chorales_test"
MIDI_NOTTINGHAM_TRAIN = "midi_nottingham_train"
MIDI_NOTTINGHAM_VALID = "midi_nottingham_valid"
MIDI_NOTTINGHAM_TEST = "midi_nottingham_test"
MIDI_MUSEDATA_TRAIN = "midi_musedata_train"
MIDI_MUSEDATA_VALID = "midi_musedata_valid"
MIDI_MUSEDATA_TEST = "midi_musedata_test"

TIMIT = "timit"
TIMIT_TRAIN = "timit_train"
TIMIT_TEST = "timit_test"


DATAPATHS_MAPPING = {
    LIBRISPEECH_TRAIN: os.path.join(SOURCE_DIRECTORY, "librispeech", "train.txt"),
    LIBRISPEECH_TRAIN_CLEAN_100: os.path.join(SOURCE_DIRECTORY, "librispeech", "train-clean-100.txt"),
    LIBRISPEECH_TRAIN_CLEAN_360: os.path.join(SOURCE_DIRECTORY, "librispeech", "train-clean-360.txt"),
    LIBRISPEECH_TRAIN_OTHER_500: os.path.join(SOURCE_DIRECTORY, "librispeech", "train-other-500.txt"),
    LIBRISPEECH_DEV_CLEAN: os.path.join(SOURCE_DIRECTORY, "librispeech", "dev-clean.txt"),
    LIBRISPEECH_DEV_OTHER: os.path.join(SOURCE_DIRECTORY, "librispeech", "dev-other.txt"),
    LIBRISPEECH_TEST_CLEAN: os.path.join(SOURCE_DIRECTORY, "librispeech", "test-clean.txt"),
    LIBRISPEECH_TEST_OTHER: os.path.join(SOURCE_DIRECTORY, "librispeech", "test-other.txt"),

    LIBRISPEECH_LM: os.path.join(SOURCE_DIRECTORY, "librispeech_lm", "librispeech_lm.txt"),

    BILLION_TRAINING: os.path.join(SOURCE_DIRECTORY, "billion", "training.txt"),
    BILLION_HELDOUT: os.path.join(SOURCE_DIRECTORY, "billion", "heldout.txt"),

    PENN_TREEBANK_TRAIN: os.path.join(SOURCE_DIRECTORY, "penn_treebank", "train.txt"),
    PENN_TREEBANK_VALID: os.path.join(SOURCE_DIRECTORY, "penn_treebank", "valid.txt"),
    PENN_TREEBANK_TEST: os.path.join(SOURCE_DIRECTORY, "penn_treebank", "test.txt"),

    MIDI_PIANO_TRAIN: os.path.join(SOURCE_DIRECTORY, "midi", "piano", "train.txt"),
    MIDI_PIANO_VALID: os.path.join(SOURCE_DIRECTORY, "midi", "piano", "valid.txt"),
    MIDI_PIANO_TEST: os.path.join(SOURCE_DIRECTORY, "midi", "piano", "test.txt"),
    MIDI_CHORALES_TRAIN: os.path.join(SOURCE_DIRECTORY, "midi", "jsb_chorales", "train.txt"),
    MIDI_CHORALES_VALID: os.path.join(SOURCE_DIRECTORY, "midi", "jsb_chorales", "valid.txt"),
    MIDI_CHORALES_TEST: os.path.join(SOURCE_DIRECTORY, "midi", "jsb_chorales", "test.txt"),
    MIDI_NOTTINGHAM_TRAIN: os.path.join(SOURCE_DIRECTORY, "midi", "nottingham", "train.txt"),
    MIDI_NOTTINGHAM_VALID: os.path.join(SOURCE_DIRECTORY, "midi", "nottingham", "valid.txt"),
    MIDI_NOTTINGHAM_TEST: os.path.join(SOURCE_DIRECTORY, "midi", "nottingham", "test.txt"),
    MIDI_MUSEDATA_TRAIN: os.path.join(SOURCE_DIRECTORY, "midi", "muse_data", "train.txt"),
    MIDI_MUSEDATA_VALID: os.path.join(SOURCE_DIRECTORY, "midi", "muse_data", "valid.txt"),
    MIDI_MUSEDATA_TEST: os.path.join(SOURCE_DIRECTORY, "midi", "muse_data", "test.txt"),

    TIMIT_TRAIN: os.path.join(SOURCE_DIRECTORY, "timit", "train.txt"),
    TIMIT_TEST: os.path.join(SOURCE_DIRECTORY, "timit", "test.txt")
}


DATASETS = SimpleNamespace(
    TIMIT=SimpleNamespace(train=TIMIT_TRAIN, test=TIMIT_TEST),
    MIDI_PIANO=SimpleNamespace(train=MIDI_PIANO_TRAIN, valid=MIDI_PIANO_VALID, test=MIDI_PIANO_TEST),
)
