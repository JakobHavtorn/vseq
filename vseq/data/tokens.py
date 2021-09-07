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

# inferred from auto-aligned libri train data (960 hours)
# may not be the full phoneset. Could be this: https://en.wikipedia.org/wiki/ARPABET
# The alignments are taken from this paper: https://arxiv.org/pdf/1904.03670.pdf
# And created with this tool: https://montrealcorpustools.github.io/Montreal-Forced-Aligner/images/MFA_paper_Interspeech2017.pdf
LIBRI_PHONESET_INFER = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH',
                        'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH',
                        'UW', 'V', 'W', 'Y', 'Z', 'ZH'] 
PHN_SIL = 'sil'
LIBRI_PHONESET_SPECIAL = ['', 'sil', 'sp', 'spn']
