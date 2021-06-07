import argparse
import os

from numpy import source

from vseq.settings import SOURCE_DIRECTORY

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", "-d", type=str, help="Directory to search for files")
parser.add_argument("--include_ext", "-i", type=str, nargs="+", help="Extra extensions to include")
parser.add_argument("--source_name", "-s", type=str, help="Name of the source file")

args = parser.parse_args()

source_file_path = os.path.join(SOURCE_DIRECTORY, args.source_name)


if not os.path.exists(args.data_dir):
    raise FileExistsError(f"Path does not exist {args.data_dir}")


source_files = []

for root, dirs, files in os.walk(args.data_dir):
    for f in files:
        name, ext = os.path.splitext(f)
        if ext in args.include_ext:
            path = os.path.join(root, name)
            source_files.append(path)


print(f"Found {len(source_files)} source files of which {len(set(source_files))} have unique names (without extension)")

header = "filename"
source_file_lines = [header]
source_file_lines.extend(source_files)
source_file_content = "\n".join(source_file_lines)


os.makedirs(os.path.dirname(source_file_path), exist_ok=True)
with open(source_file_path, "w") as source_file_buffer:
    source_file_buffer.write(source_file_content)

print(f"Created source file - {source_file_path}")
