import random
import time
import sys


def init_print_rows(world_size):
    for line in reversed(range(world_size)):
        print(f"Line {line}")
        time.sleep(.25)


def print_to_line(s, rank, world_size):
    lines_up = f"\033[{rank + 1}F"
    lines_down = f"\033[{rank}E"
    end = '' if rank == 0 else '\n'
    print(f"{lines_up}{s} with {end=}{lines_down}", end=end)
    sys.stdout.flush()


world_size = 5

random.seed(0)

lines = [random.randint(0, world_size - 1) for _ in range(100)]

print(lines)

init_print_rows(world_size)

for i, l in enumerate(lines):
    print_to_line(f"Line {l} (update {i})", l, world_size)
    time.sleep(0.5)

print()
