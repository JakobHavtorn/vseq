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

device = get_free_gpus(1, require_unused=True)
# device = "cpu"
print(device)

# custom VS regular pytorch module
model = LSTM(in_features, out_features, num_layers=num_layers, bidirectional=bidirectional, layer_norm=False, jit_compile=True).to(device)
nn_model = nn.LSTM(in_features, out_features, num_layers=num_layers, bidirectional=bidirectional).to(device)

# jit'ed module
model_jit = torch.jit.script(model)


def get_x():
    x = torch.randn(seq_len, batch_size, in_features).to(device)
    return x


def get_state():
    state = (
        torch.randn(num_layers * num_directions, batch_size, out_features, device=device),
        torch.randn(num_layers * num_directions, batch_size, out_features, device=device),
    )
    return state


torch.manual_seed(42)
print(f"forward()            | nn.LSTM:         {timeit('nn_model(get_x(), get_state())[0].sum().item()', globals=globals(), number=10, repeats=50)}")
torch.manual_seed(42)
print(f"forward()            | jit custom LSTM: {timeit('model_jit(get_x(), get_state())[0].sum().item()', globals=globals(), number=10, repeats=50)}")
torch.manual_seed(42)
print(f"forward()            | custom LSTM:     {timeit('model(get_x(), get_state())[0].sum().item()', globals=globals(), number=10, repeats=50)}")

torch.manual_seed(42)
print(f"forward()+backward() | nn.LSTM:         {timeit('nn_model(get_x(), get_state())[0].sum().backward()', globals=globals(), number=10, repeats=50)}")
torch.manual_seed(42)
print(f"forward()+backward() | jit custom LSTM: {timeit('model_jit(get_x(), get_state())[0].sum().backward()', globals=globals(), number=10, repeats=50)}")
torch.manual_seed(42)
print(f"forward()+backward() | custom LSTM:     {timeit('model(get_x(), get_state())[0].sum().backward()', globals=globals(), number=10, repeats=50)}")

# forward()            | nn.LSTM:          namespace(min=0.007419191561639309, max=0.010110171288251877, mean=0.00823103709369898, median=0.007983970679342747, std=0.0006925272679852692, number=50, repeats=50)
# forward()            | custom LSTM:      namespace(min=0.022613052278757095, max=0.040045680664479735, mean=0.029453871361911296, median=0.02876217933371663, std=0.004078533439866274, number=10, repeats=50)
# forward()            | jit custom LSTM:  namespace(min=0.015540992096066474, max=0.02707811463624239, mean=0.020774183165282012, median=0.020505926478654145, std=0.0026901405449034, number=10, repeats=50)
# forward()+backward() | nn.LSTM:          namespace(min=0.012307688686996698, max=0.017329868767410515, mean=0.014820898579433561, median=0.014775047730654477, std=0.0011021714492689023, number=20, repeats=50)
# forward()+backward() | custom LSTM:      namespace(min=0.05178355723619461, max=0.09266387820243835, mean=0.07213428250700235, median=0.07168622519820929, std=0.00846900343022748, number=5, repeats=50)
# forward()+backward() | jit custom LSTM:  namespace(min=0.02761768363416195, max=0.1429811492562294, mean=0.06067255970090628, median=0.06149108987301588, std=0.021960204424152672, number=1, repeats=50)
