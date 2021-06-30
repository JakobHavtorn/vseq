import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

from vseq.utils.device import get_free_gpus
from vseq.utils.timing import timeit

import uniplot


BATCH_SIZE = 4
TIME_STEPS_MULTIPLIER = 100
sizes = [16, 64, 256, 512, 1024, 2048, 4096]

try:
    device = get_free_gpus(1)
except Exception:
    print("Running on CPU since no GPUs available.")
    device = "cpu"


median_time_linear = []
median_time_conv1d = []

for size in tqdm(sizes):
    linear = nn.Linear(size, size).to(device)
    conv1d = nn.Conv1d(1, size, kernel_size=size, stride=size).to(device)

    conv1d.weight.data = linear.weight.data.view(*conv1d.weight.shape)
    conv1d.bias.data = linear.bias.data.view(*conv1d.bias.shape)

    x_linear = torch.ones(BATCH_SIZE, TIME_STEPS_MULTIPLIER, size, device=device)
    x_conv1d = torch.ones(BATCH_SIZE, 1, TIME_STEPS_MULTIPLIER * size, device=device)

    y_linear = linear(x_linear)
    y_conv1d = conv1d(x_conv1d)

    np.testing.assert_allclose(
        y_linear.detach().cpu().numpy(),
        y_conv1d.view(BATCH_SIZE, size, TIME_STEPS_MULTIPLIER).detach().permute(0,2,1).cpu().numpy(),
        rtol=1e-5,
        atol=1e-5
    )

    timings_linear = timeit("linear(x_linear)[0,0,0].item()", globals=globals(), print_results=True)
    timings_conv1d = timeit("conv1d(x_conv1d)[0,0,0].item()", globals=globals(), print_results=True)

    median_time_linear.append(timings_linear.median * 1000)
    median_time_conv1d.append(timings_conv1d.median * 1000)


uniplot.plot([median_time_linear, median_time_conv1d], [sizes, sizes], legend_labels=["linear", "conv1d"])
uniplot.plot([median_time_linear, median_time_conv1d], [np.log2(sizes), np.log2(sizes)], legend_labels=["linear", "conv1d"])
uniplot.plot([np.log(median_time_linear), np.log(median_time_conv1d)], [sizes, sizes], legend_labels=["linear", "conv1d"])
uniplot.plot([np.log(median_time_linear), np.log(median_time_conv1d)], [np.log2(sizes), np.log2(sizes)], legend_labels=["linear", "conv1d"])
