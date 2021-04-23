import argparse

from vseq.data.text_cleaners import clean_librispeech
from vseq.data.vocabulary import build_voabulary


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="ptb_train", type=int, help="dataset")
args = parser.parse_args()


build_voabulary(args.dataset, cleaner_fcn=clean_librispeech)
