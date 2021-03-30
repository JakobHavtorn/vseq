import os

import vseq.settings


LIBRISPEECH_TRAIN = 'libri_train'
LIBRISPEECH_TRAIN_CLEAN_100 = 'libri_train_clean_100'
LIBRISPEECH_TRAIN_CLEAN_360 = 'libri_train_clean_360'
LIBRISPEECH_TRAIN_OTHER_500 = 'libri_train_other_500'
LIBRISPEECH_DEV_CLEAN = 'libri_dev_clean'
LIBRISPEECH_DEV_OTHER = 'libri_dev_other'
LIBRISPEECH_TEST_CLEAN = 'libri_test_clean'
LIBRISPEECH_TEST_OTHER = 'libri_test_other'

PENN_TREEBANK_TRAIN = 'ptb_train'
PENN_TREEBANK_VALID = 'ptb_valid'
PENN_TREEBANK_TEST = 'ptb_test'


DATAPATHS_MAPPING = {
    LIBRISPEECH_TRAIN: os.path.join(vseq.settings.DATA_DIRECTORY, 'librispeech', 'train.txt'),
    LIBRISPEECH_TRAIN_CLEAN_100: os.path.join(vseq.settings.DATA_DIRECTORY, 'librispeech', 'train-clean-100.txt'),
    LIBRISPEECH_TRAIN_CLEAN_360: os.path.join(vseq.settings.DATA_DIRECTORY, 'librispeech', 'train-clean-360.txt'),
    LIBRISPEECH_TRAIN_OTHER_500: os.path.join(vseq.settings.DATA_DIRECTORY, 'librispeech', 'train-other-500.txt'),
    LIBRISPEECH_DEV_CLEAN: os.path.join(vseq.settings.DATA_DIRECTORY, 'librispeech', 'dev-clean.txt'),
    LIBRISPEECH_DEV_OTHER: os.path.join(vseq.settings.DATA_DIRECTORY, 'librispeech', 'dev-other.txt'),
    LIBRISPEECH_TEST_CLEAN: os.path.join(vseq.settings.DATA_DIRECTORY, 'librispeech', 'test-clean.txt'),
    LIBRISPEECH_TEST_OTHER: os.path.join(vseq.settings.DATA_DIRECTORY, 'librispeech', 'test-other.txt'),
    PENN_TREEBANK_TRAIN: os.path.join(vseq.settings.DATA_DIRECTORY, 'penn_treebank', 'train.txt'),
    PENN_TREEBANK_VALID: os.path.join(vseq.settings.DATA_DIRECTORY, 'penn_treebank', 'valid.txt'),
    PENN_TREEBANK_TEST: os.path.join(vseq.settings.DATA_DIRECTORY, 'penn_treebank', 'test.txt'),
}
