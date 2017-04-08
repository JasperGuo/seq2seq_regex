# coding=utf8

"""
Extract Samples from predictions for analysis
"""
import argparse
import os
import re
import json

ANALYSIS_PATH = "analysis.txt"

STRING_MATCH_PATTERN = re.compile(r'exact_match_diff: (\d)\n')
DFA_PATTERN = re.compile(r'dfa_equality: (\d)\n')
SENTENCE_PATTERN = re.compile(r'S: (.*)\n')
PREDICTION_PATTERN = re.compile(r'p: (.*)\n')
GROUND_TRUTH_PATTERN = re.compile(r'T: (.*)\n')


def read_analysis_file():
    analysis_list = list()
    with open(ANALYSIS_PATH, "r") as f:
        for line in f:
            analysis_list.append(line.strip())
    return analysis_list


def extract_from_text_file(path, target_list):
    """
    :param target_list:
    :param path:
    :return:
    """
    new_dir = os.path.join(path, "extracted")
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    for file in os.listdir(path):
        file_name = os.path.join(path, file)
        if not os.path.isfile(file_name):
            continue
        result = dict()
        with open(file_name, "r") as f:
            line = f.readline()
            while line and line != "":
                match = STRING_MATCH_PATTERN.match(line)
                if match:
                    score = int(match.group(1).strip())
                    dfa_equality = int(DFA_PATTERN.match(f.readline()).group(1).strip())
                    sentence = SENTENCE_PATTERN.match(f.readline()).group(1).strip()
                    if sentence in target_list:

                        if sentence not in result:
                            result[sentence] = {
                                "detail": list(),
                                "dfa": list(),
                                "string": list()
                            }
                        result[sentence]["detail"].append({
                            "dfa_equality": dfa_equality,
                            "score": score,
                            "sentence": sentence,
                            "prediction": PREDICTION_PATTERN.match(f.readline()).group(1).strip(),
                            "truth": GROUND_TRUTH_PATTERN.match(f.readline()).group(1).strip()
                        })
                        result[sentence]["dfa"].append(dfa_equality)
                        result[sentence]["string"].append(score)
                line = f.readline()
        new_file_name = os.path.join(new_dir, file)
        with open(new_file_name, "w") as f:
            f.write(json.dumps(result, indent=4))


def main(path):
    analysis_list = read_analysis_file()
    print(analysis_list)
    extract_from_text_file(path, analysis_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate DFA Equality')
    parser.add_argument('--path', help='epoch file', required=True)

    args = parser.parse_args()
    main(args.path)
