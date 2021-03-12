import os

import vseq


LIBRISPEECH_TRAIN = 'libri_train'
LIBRISPEECH_DEV_CLEAN = 'libri_dev_clean'
LIBRISPEECH_DEV_OTHER = 'libri_dev_other'
LIBRISPEECH_TEST_CLEAN = 'libri_test_clean'
LIBRISPEECH_TEST_OTHER = 'libri_test_other'


DATAPATHS_MAPPING = {
    LIBRISPEECH_TRAIN: os.path.join(vseq.settings.DATA_DIRECTORY, 'librispeech', 'train.txt'),
    LIBRISPEECH_DEV_CLEAN: os.path.join(vseq.settings.DATA_DIRECTORY, 'librispeech', 'dev_clean.txt'),
    LIBRISPEECH_DEV_OTHER: os.path.join(vseq.settings.DATA_DIRECTORY, 'librispeech', 'dev_other.txt'),
    LIBRISPEECH_TEST_CLEAN: os.path.join(vseq.settings.DATA_DIRECTORY, 'librispeech', 'test_clean.txt'),
    LIBRISPEECH_TEST_OTHER: os.path.join(vseq.settings.DATA_DIRECTORY, 'librispeech', 'test_other.txt'),
}
