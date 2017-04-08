# coding=utf8

"""
Recalculate Beam Search Accuracy
"""

# coding=utf8

import argparse
import re
import os

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


def process(file_path):
    predict_results = read(file_path)

    total = len(predict_results.keys())
    accuracy, dfa_accuracy = 0, 0
    for key, value in predict_results.items():
        if sum(value["dfa"]) > 0:
            dfa_accuracy += 1
        if sum(value["string"]) > 0:
            accuracy += 1

    accuracy /= total
    dfa_accuracy /= total

    head, tile = os.path.split(file_path)

    log_file = os.path.join(head, "recalc_dfa.txt")

    with open(log_file, "a") as f:
        f.write("%s, accuracy: %f, dfa_accuracy: %f\n" % (tile, accuracy, dfa_accuracy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate DFA Equality')
    parser.add_argument('--file', help='epoch file', required=True)

    args = parser.parse_args()
    process(args.file)


