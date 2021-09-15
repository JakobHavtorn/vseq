import os
import subprocess
import re

from io import StringIO
from typing import Optional, Union, List

import torch
import pandas as pd
import numpy as np


def get_visible_devices_global_ids():
    """Return the global indices of the visible devices.

    If `CUDA_VISIBLE_DEVICES` is not set, returns all devices.
    If it is set to the empty string, return no devices.
    """
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        return list(range(torch.cuda.device_count()))

    if os.environ["CUDA_VISIBLE_DEVICES"] == "":
        return []

    visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
    visible_devices = re.split("; |, ", visible_devices)
    visible_devices = sorted([int(idx) for idx in visible_devices])
    return visible_devices


def get_gpu_memory_usage(do_print: bool = False) -> pd.DataFrame:
    """Return the free and used memory per GPU device on the node"""
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])

    gpu_df = pd.read_csv(StringIO(gpu_stats.decode("utf-8")), names=["memory.used", "memory.free"], skiprows=1)

    gpu_df.rename(columns={"memory.used": "used", "memory.free": "free"}, inplace=True)
    gpu_df["free"] = gpu_df["free"].map(lambda x: int(x.rstrip(" [MiB]")))
    gpu_df["used"] = gpu_df["used"].map(lambda x: int(x.rstrip(" [MiB]")))

    if do_print:
        print("GPU usage (MB):\n{}".format(gpu_df))
    return gpu_df


def get_free_gpus(
    n_gpus: int = 1, require_unused: bool = True, fallback_to_cpu: bool = True
) -> Union[torch.device, List[torch.device]]:
    """Return one or more available/visible (and unused) devices giving preference to those with most free memory"""
    gpu_df = get_gpu_memory_usage()

    visible_devices_global_ids = get_visible_devices_global_ids()

    gpu_df = gpu_df[gpu_df.index.isin(visible_devices_global_ids)]

    if require_unused:
        gpu_df = gpu_df[gpu_df.used < 10]

    gpu_df = gpu_df.sort_values(by="free", ascending=False)

    global_device_ids = gpu_df.iloc[:n_gpus].index.to_list()
    local_device_idx = [visible_devices_global_ids.index(device_id) for device_id in global_device_ids]
    devices = [torch.device(idx) for idx in local_device_idx]

    if len(devices) < n_gpus:
        raise RuntimeError(f"Found {len(devices)} (free) GPUs but required {n_gpus}")

    if len(devices) > 0:
        print(f"Selected global device(s): {global_device_ids}")
        return devices[0] if len(devices) == 1 else devices

    if fallback_to_cpu:
        return torch.device("cpu")

    raise RuntimeError(f"Found {len(devices)} (free) GPUs. If you want to fall back to CPU, set 'fallback_to_cpu=True'")


def get_device(idx: Optional[int] = None):
    """Return the device to run on (cpu or cuda).

    If `CUDA_VISIBLE_DEVICES` is not set we assume that no devices are wanted and return the CPU.
    This is contrary to standard `torch.cuda.is_available()` behaviour

    If idx is specified, return the GPU corresponding to that index in the local scope.
    """
    if not torch.cuda.is_available() or "CUDA_VISIBLE_DEVICES" not in os.environ:
        return torch.device("cpu")

    if idx is None:
        return torch.device("cuda:0")

    local_device_indices = list(range(torch.cuda.device_count()))
    return torch.device(f"cuda:{local_device_indices[idx]}")


def test_gpu_functionality():
    """Returns `True` if a GPU is available and functionality is OK, otherwise raises an error"""
    # Set GPU as the device if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print()

    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), "GB")
        print("CUDA version:", torch.version.cuda)
        torch.zeros(1).cuda()
        return True
    else:
        # provoke an error
        torch.zeros(1).cuda()


if __name__ == "__main__":
    test_gpu_functionality()
    free_gpu_id = get_free_gpus()
    print(free_gpu_id)
