# coding=utf8


import argparse
from util import save_data, read_data


def _replace(case):
    _case = case.replace("dog", '<M0>')
    _case = _case.replace("truck", '<M1>')
    _case = _case.replace("ring", '<M2>')
    _case = _case.replace("lake", '<M3>')
    return _case


def replace(file_path, target_path):
    data = read_data(file_path)

    for sample in data:
        sample["case"] = _replace(sample["case"])

    save_data(target_path, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", help="source file")
    parser.add_argument("--target", help="target file")

    args = parser.parse_args()

    replace(args.source, args.target)
