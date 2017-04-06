# coding=utf8

"""
Get Error From Epoch Log
"""
import argparse
import os
import re
from pprint import pprint
import json

STRING_MATCH_PATTERN = re.compile(r'exact_match_diff: (\d)\n')
DFA_PATTERN = re.compile(r'dfa_equality: (\d)\n')
SENTENCE_PATTERN = re.compile(r'S: (.*)\n')
PREDICTION_PATTERN = re.compile(r'p: (.*)\n')
GROUND_TRUTH_PATTERN = re.compile(r'T: (.*)\n')


def calc_err(path):
    """
    :param format:
    :param sentence:
    :param path:
    :return:
    """
    result = list()
    with open(path, "r") as f:
        line = f.readline()
        while line and line != "":
            match = STRING_MATCH_PATTERN.match(line)
            if match:
                score = int(match.group(1).strip())
                dfa_equality = int(DFA_PATTERN.match(f.readline()).group(1).strip())
                se = SENTENCE_PATTERN.match(f.readline()).group(1).strip()
                pred = PREDICTION_PATTERN.match(f.readline()).group(1).strip()
                truth = GROUND_TRUTH_PATTERN.match(f.readline()).group(1).strip()
                if dfa_equality == 0:
                    result.append({
                        "prediction": pred,
                        "truth": truth,
                        "sentence": se
                    })
            line = f.readline()

    head, tile = os.path.split(path)
    result_path = os.path.join(head, "error_" + tile)
    with open(result_path, "w") as f:
        f.write(json.dumps(result, indent=4))
    pprint(result)
    print(len(result))
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate DFA Equality')
    parser.add_argument('--path', help='epoch file', required=True)

    args = parser.parse_args()
    calc_err(args.path)
