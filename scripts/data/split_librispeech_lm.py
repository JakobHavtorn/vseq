
BP = "/mnt/data/research/source/librispeech_lm"
FP = f"{BP}/librispeech_lm.txt"

with open(FP, "r") as f:
    lines = f.read().split("\n")

header = lines[0]

for i in range(1, len(lines)):
    new_source = f"{header}\n{lines[i]}"
    new_source_path = f"{BP}/librispeech_lm_{i-1}.txt"
    with open(new_source_path, "w") as f:
        f.write(new_source)
    