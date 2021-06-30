
def set_dropout(model, p):
    if hasattr(model, 'p'):
        model.p = p
    for submodule in model.children():
        set_dropout(submodule, p)