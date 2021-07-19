from tqdm import tqdm
import os

FP = "/data/research/source/librispeech/train.txt"

with open(FP, "r") as f:
    lines = f.read().split("\n")

lines[0] = lines[0] + ",length.txt.chars\n"

for i in tqdm(range(1, len(lines))):
    tf = lines[i].split(",")[0] + ".txt"
    with open(tf, "r") as f:
        num_chars = len(f.read())
    
    nl = f"{lines[i]},{num_chars}\n"
    lines[i] = nl

lines[-1] = lines[-1].strip()

os.remove(FP)

with open(FP, "w") as buffer:
    buffer.writelines(lines)

