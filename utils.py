import os

INPUT_PATH = "input"
OUTPUT_PATH = "output"

def in_file(filename):
    return os.path.join(INPUT_PATH,filename)

def out_file(filename):
    return os.path.join(OUTPUT_PATH,filename)

