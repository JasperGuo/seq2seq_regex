# coding=utf8

import argparse
from util import read_data, save_data


def construct_data(file_path, target_path):
    raw_data = read_data(file_path)

    data = list()

    for sample in raw_data:
        sentence = sample["sentence"]
        regex = sample["regex"]
        positive_cases = sample["cases"]["positive"]
        negative_cases = sample["cases"]["negative"]
        for case in positive_cases:
            data.append({
                "sentence": sentence,
                "case": case,
                "label": 1,
                "regex": regex
            })
        for case in negative_cases:
            data.append({
                "sentence": sentence,
                "case": case,
                "label": 0,
                "regex": regex
            })

    save_data(target_path, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", help="source file")
    parser.add_argument("--target", help="target file")
    args = parser.parse_args()

    construct_data(args.source, args.target)
