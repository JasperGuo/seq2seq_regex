# coding=utf8

"""
Get Analysis Samples in test set
"""

import json
import random

PATH = "dataset\\test.json"
TARGET_PATH = "analysis.txt"


def main():
    with open(PATH, "r") as f:
        test_set = json.load(f)

    random.shuffle(test_set)
    samples = random.sample(test_set, 50)

    with open(TARGET_PATH, "w") as f:
        for sample in samples:
            f.write(sample["sentence"] + "\n")


if __name__ == "__main__":
    main()