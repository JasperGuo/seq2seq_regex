# coding=utf8

"""
Refine Regex Expression test case, shorten the random string
"""
import argparse
from multiprocessing import Process
from util import save_data, read_data
import re


def _replace(regex):
    regex = regex.replace("<VOW>", 'AEIOUaeiou')
    regex = regex.replace("<NUM>", '0-9')
    regex = regex.replace("<LET>", 'A-Za-z')
    regex = regex.replace("<CAP>", 'A-Z')
    regex = regex.replace("<LOW>", 'a-z')
    regex = regex.replace("<M0>", 'dog')
    regex = regex.replace("<M1>", 'truck')
    regex = regex.replace("<M2>", 'ring')
    regex = regex.replace("<M3>", 'lake')
    regex = regex.replace(" ", "")
    return regex

MATCH = "MATCH"
NOT_MATCH = "NOT_MATCH"


def worker(regex_pattern, string):
    match = regex_pattern.findall(string)

    if match:
        return MATCH
    return NOT_MATCH


def shorten_test_case(file_path, target_path, error_path, max_len, timeout):
    original_data = read_data(file_path)
    processed_data = list()
    fail_list = list()
    for sample in original_data:
        case = sample["case"]
        if len(case) <= max_len:
            processed_data.append(sample)
        else:
            length = len(case)
            parts = round(length/max_len)
            pattern = re.compile(_replace(sample["regex"]))
            for i in range(parts):
                string = case[i*max_len: (i+1)*max_len]

                _process = Process(target=worker, args=(pattern, string,))
                _process.start()
                _process.join(timeout=timeout)
                _process.terminate()

                if _process.exitcode == MATCH and sample["label"] == 1:
                    sample["case"] = string
                    processed_data.append(sample)
                    break
                elif (_process.exitcode == NOT_MATCH) and sample["label"] == 0:
                    sample["case"] = string
                    processed_data.append(sample)
                    break
            else:
                # Fail to shorten the case
                fail_list.append(sample)
                print(sample["regex"], case)

    save_data(target_path, processed_data)
    save_data(error_path, fail_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", help="source file")
    parser.add_argument("--target", help="target file")
    parser.add_argument("--error", help="error file")
    parser.add_argument("--length", help="case max length")
    parser.add_argument("--timeout", help="timeout")

    args = parser.parse_args()

    shorten_test_case(args.source, args.target, args.error, int(args.length), int(args.timeout))
