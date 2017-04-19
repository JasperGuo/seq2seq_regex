# coding=utf8
import argparse

from util import read_data, save_data


def split(file_path, target_path):
    raw_data = read_data(file_path)

    for sample in raw_data:
        case = sample["case"]
        case = case.replace(" ", "")
        if "<M0>" in case:
            case = case.replace("<M0>", " <M0> ")
        if "<M1>" in case:
            case = case.replace("<M1>", " <M1> ")
        if "<M2>" in case:
            case = case.replace("<M2>", " <M2> ")
        if "<M3>" in case:
            case = case.replace("<M3>", " <M3> ")

        i = 0
        length = len(case)
        new_string = ""
        while i < length:
            if str.isspace(case[i]):
                j = i+1
                while (not str.isspace(case[j])) and j < length:
                    j += 1
                word = case[i+1:j]
                i = j + 1
            else:
                word = case[i]
                i += 1
            new_string += " " + word
        sample["case"] = new_string.strip()

    result = list()
    for sample in raw_data:
        case = sample["case"]
        if case and len(case.replace(" ", "")) == 0:
            continue
        result.append(sample)

    save_data(target_path, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", help="source file")
    parser.add_argument("--target", help="target file")
    args = parser.parse_args()

    split(args.source, args.target)
