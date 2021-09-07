import os

for i in range(-1, -51, -1):
    os.system(f"""CUDA_VISIBLE_DEVICES=6 python experiment_w2v2_context_probe.py --label_shift {i}""")