import math

from typing import List

import torch
import torch.nn as nn

from vseq.modules.custom_recurrent import LSTM
from vseq.utils.device import get_free_gpus
from vseq.utils.timing import timeit


seq_len = 50
batch_size = 64
in_features = 464
out_features = 373
num_directions = 1
num_layers = 1

bidirectional = num_directions > 1

device = get_free_gpus(1, require_unused=False)
# device = "cpu"
print(device)

# custom VS regular pytorch module
model = LSTM(in_features, out_features, num_layers=num_layers, bidirectional=bidirectional, layer_norm=True).to(device)
model_native = nn.LSTM(in_features, out_features, num_layers=num_layers, bidirectional=bidirectional).to(device)

# jit'ed module
model_jit = torch.jit.script(model)

# saved jit'ed module
torch.jit.save(model_jit, "model_jit.pt")
model_jit_saved = torch.jit.load("model_jit.pt", map_location=device)


def get_x():
    # seq_len = torch.randint(low=50, high=150, size=(1,))
    x = torch.randn(seq_len, batch_size, in_features).to(device)
    return x


def get_state():
    state = (
        torch.randn(num_layers * num_directions, batch_size, out_features, device=device),
        torch.randn(num_layers * num_directions, batch_size, out_features, device=device),
    )
    return state


print(f"forward()            | native model: {timeit('model_native(get_x(), get_state())', globals=globals(), repeats=50)}")
print(f"forward()            | nn model:     {timeit('model(get_x(), get_state())', globals=globals(), repeats=50)}")
print(f"forward()            | jit model:    {timeit('model_jit(get_x(), get_state())', globals=globals(), repeats=50)}")
print(f"forward()            | saved model:  {timeit('model_jit_saved(get_x(), get_state())', globals=globals(), repeats=50)}")

print(f"forward()+backward() | native model: {timeit('model_native(get_x(), get_state())[0].sum().backward()', globals=globals(), repeats=50)}")
print(f"forward()+backward() | nn model:     {timeit('model(get_x(), get_state())[0].sum().backward()', globals=globals(), repeats=50)}")
print(f"forward()+backward() | jit model:    {timeit('model_jit(get_x(), get_state())[0].sum().backward()', globals=globals(), repeats=50)}")
print(f"forward()+backward() | saved model:  {timeit('model_jit_saved(get_x(), get_state())[0].sum().backward()', globals=globals(), repeats=50)}")

# forward()            | native model: namespace(min=0.007419191561639309, max=0.010110171288251877, mean=0.00823103709369898, median=0.007983970679342747, std=0.0006925272679852692, number=50, repeats=50)
# forward()            | nn model:     namespace(min=0.022613052278757095, max=0.040045680664479735, mean=0.029453871361911296, median=0.02876217933371663, std=0.004078533439866274, number=10, repeats=50)
# forward()            | jit model:    namespace(min=0.015540992096066474, max=0.02707811463624239, mean=0.020774183165282012, median=0.020505926478654145, std=0.0026901405449034, number=10, repeats=50)
# forward()            | saved model:  namespace(min=0.014788175560534, max=0.026156367734074593, mean=0.019867969390004875, median=0.019958647619932893, std=0.0018888938889088237, number=10, repeats=50)
# forward()+backward() | native model: namespace(min=0.012307688686996698, max=0.017329868767410515, mean=0.014820898579433561, median=0.014775047730654477, std=0.0011021714492689023, number=20, repeats=50)
# forward()+backward() | nn model:     namespace(min=0.05178355723619461, max=0.09266387820243835, mean=0.07213428250700235, median=0.07168622519820929, std=0.00846900343022748, number=5, repeats=50)
# forward()+backward() | jit model:    namespace(min=0.02761768363416195, max=0.1429811492562294, mean=0.06067255970090628, median=0.06149108987301588, std=0.021960204424152672, number=1, repeats=50)
# forward()+backward() | saved model:  namespace(min=0.029059674590826035, max=0.08644191175699234, mean=0.050898110941052434, median=0.04758156277239323, std=0.014786397951337588, number=1, repeats=50)