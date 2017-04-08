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


def read(file_path):
    result = dict()
    file_name = os.path.basename(file_path)
    with open(file_path, "r") as f:
        line = f.readline()
        while line and line != "":
            match = STRING_MATCH_PATTERN.match(line)
            if match:
                score = int(match.group(1).strip())
                dfa_equality = int(DFA_PATTERN.match(f.readline()).group(1).strip())
                sentence = SENTENCE_PATTERN.match(f.readline()).group(1).strip()

                if sentence not in result:
                    result[sentence] = {
                        "detail": [],
                        "dfa": [],
                        "string": []
                    }
                result[sentence]["detail"].append({
                    "file_name": '_'.join(["dfa", file_name]),
                    "dfa_equality": dfa_equality,
                    "score": score,
                    "prediction": PREDICTION_PATTERN.match(f.readline()).group(1).strip(),
                    "truth": GROUND_TRUTH_PATTERN.match(f.readline()).group(1).strip()
                })
                result[sentence]["dfa"].append(dfa_equality)
                result[sentence]["string"].append(score)
            line = f.readline()
    return result


def calc_err(path):
    """
    :param format:
    :param sentence:
    :param path:
    :return:
    """
    all_predictions = read(path)

    errors = list()

    for sentence, value in all_predictions.items():
        if sum(value["dfa"]) == 0:
            errors.append({
                "sentence": sentence,
                "detail": value["detail"]
            })

    head, tile = os.path.split(path)
    result_path = os.path.join(head, "error_" + tile)
    with open(result_path, "w") as f:
        f.write(json.dumps(errors, indent=4))
    # pprint(errors)
    print(len(errors))
    return errors

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate DFA Equality')
    parser.add_argument('--path', help='epoch file', required=True)

    args = parser.parse_args()
    calc_err(args.path)
