import vseq.models


def get_model(name):
    return getattr(vseq.models, name)
