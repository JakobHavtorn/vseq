import re


def clean_librispeech(txt):
    """
    Lowercases the input string and removes anything but characters a-z and whitespaces.

    Also normalizes whitespace characters and strip trailing ones.

    The final character set will be:
        - letters: "a-z"
        - apostrophe: "'"
        - whitespace: " "

    Args:
        txt: Text to be normalized.

    Returns:
        str: The normalized string.
    """

    # lowercase and strip trailing whitespaces
    txt = txt.lower().strip()

    # whitespace normalization: convert whitespace sequences to a single whitespace
    txt = re.sub(r'\s+', ' ', txt)

    return txt


def clean_wsj(txt, remove_fragments=True):
    """
    Prepares WSJ transcripts according to Hannun et al., 2014:
    https://arxiv.org/pdf/1408.2873.pdf

    It is assumed that data has already been processed with Kaldi's s5 recipe.

    A full overview of the wsj transcription guidelines is found here:
    https://catalog.ldc.upenn.edu/docs/LDC93S6A/dot_spec.html

    It is not fully exhaustive which may be due to transcriber variations/mistakes.

    The final character set will be:
        - letters: "a-z"
        - noise-token: "?"
        - apostrophe: "'"
        - hyphen: "-"
        - dot: "."
        - whitespace: " "

    Args:
        txt: Text to be normalized.

    Returns:
        str: The normalized string.
    """
    txt = txt.lower()

    # Explanations for replacements:
    # - cursive apostrophe [`] should have been ' (very rare)
    # - double tilde [~~] seems to indicate silence during the full clip (very rare)
    # - noise-tag [<noise>] used to indicate noise (can be very subtle breathing between words)
    # - underscore [_] should have been a hyphen (one instance)
    # - pointy brackets [<...>] seems to indicate verbal insertion, but guidelines says deletion
    # - word enclosed in asterisk symbols [*hospital*] indicates mispronunciation, but will be spelled correctly though
    # - semi-colon [;] should have been a . in the abbreviation corp. (one instance)
    # - wrongful spelling of parentheses
    txt = txt.replace("`", "'").replace('~', '').replace('<noise>', '').replace('_', '-')
    txt = txt.replace('<', '').replace('>', '').replace('*', '').replace('corp;', 'corp.')
    txt = txt.replace('in-parenthesis', 'in-parentheses')

    # - word fragment in parenthesis [-(repub)lican] indicates missing fragment
    txt = re.sub("\([a-z'-]+\)", "", txt)

    # Everything in the remove list is vocalized punctuation, however, a few characters also have other uses:
    # - colon associated with word [securi:ty or a:] used to indicate lengthening
    # - prepended exclamation-point [!but] used for emphatic stress
    # These can, however, simply be removed anyway.
    remove = ['"', '(', ')', '{', '}', ',', '&', '/', ';', ':', '!']
    for char in remove:
        txt = txt.replace(char, '')

    # The following is also vocalized punctuation but can not simply be removed, as we sometimes want to keep these:
    # - hyphen/dash [-] when used to compound words and in the beginning/end of word fragments
    # - period [.] when used in acronyms and abbreviations
    # - single-quote ['] when used for contractions and possessive form
    txt = txt.replace('-dash', 'dash').replace('-hyphen', 'hyphen')
    txt = txt.replace('.period', 'period').replace('...ellipsis', 'ellipsis')
    txt = txt.replace("'single-quote", 'single-quote').replace('?question-mark', 'question-mark')


    if remove_fragments:
        # adjacent fragements are joined to one word
        txt = txt.replace('in- -communicado', 'incommunicado')
        txt = txt.replace('republi- -publicans', 'republicans')
        txt = txt.replace('gi- -vestigating', 'investigating')
        txt = txt.replace('exa- -cerbate', 'exacerbate')
        txt = txt.replace('rel- -linquish', 'relinquish')
        txt = txt.replace('cata- -lysmic', 'cataclysmic')
        txt = txt.replace('toro- -ronto', 'toronto')
        # all simple fragments are removed
        txt = re.sub(r"([a-z']+-( |$)|( |^)-[a-z']+)", "", txt)

    # should only be - between verbalized punctuation
    txt = txt.replace('-', '')
    # used in front of letters in acronyms and abbreviations
    txt = txt.replace('.', '')

    # whitespace normalization: convert whitespace sequences to a single whitespace
    txt = re.sub("\s+", " ", txt)

    return txt.strip()
