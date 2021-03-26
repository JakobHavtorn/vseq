import argparse

from vseq.data.text_cleaners import clean_librispeech
from vseq.data.vocabulary import build_voabulary


parser = argparse.ArgumentParser()
parser.add

build_voabulary('ptb_train', cleaner_fcn=clean_librispeech)

