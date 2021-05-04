import string

from vseq.data.datapaths import PENN_TREEBANK_TEST

START_TOKEN = "<"
END_TOKEN = ">"
DELIMITER_TOKEN = "|"
BLANK_TOKEN = "%"
UNKNOWN_TOKEN = "¿"

SPACE = " "
APOSTROPHE = "'"

ENGLISH_STANDARD = list(string.ascii_lowercase + SPACE + APOSTROPHE)
PENN_TREEBANK_ALPHABET = ENGLISH_STANDARD + [".", "-", "&", "$", "N"]
