# coding=utf8
import argparse
import os
import re
import json
from pprint import pprint

ANALYSIS_PATH = "analysis.txt"

STRING_MATCH_PATTERN = re.compile(r'exact_match_diff: (\d)\n')
DFA_PATTERN = re.compile(r'dfa_equality: (\d)\n')
SENTENCE_PATTERN = re.compile(r'S: (.*)\n')
PREDICTION_PATTERN = re.compile(r'p: (.*)\n')
GROUND_TRUTH_PATTERN = re.compile(r'T: (.*)\n')


def calc_err(path, sentence, format="json"):
    """
    :param format:
    :param sentence:
    :param path:
    :return:
    """
    print(sentence)
    result = list()
    for file in os.listdir(path):
        file_name = os.path.join(path, file)
        if not os.path.isfile(file_name):
            continue
        with open(file_name, "r") as f:
            if format == "json":
                try:
                    predictions = json.load(f)
                except:
                    continue

                for prediction in predictions:
                    se = prediction["sentence"]
                    score = prediction["score"]
                    dfa_score = int(prediction["dfa_equality"])
                    pred = prediction["prediction"]

                    if se == sentence and dfa_score == 0:
                        result.append((file, pred)),
            else:
                line = f.readline()
                while line and line != "":
                    match = STRING_MATCH_PATTERN.match(line)
                    if match:
                        score = int(match.group(1).strip())
                        dfa_equality = int(DFA_PATTERN.match(f.readline()).group(1).strip())
                        se = SENTENCE_PATTERN.match(f.readline()).group(1).strip()
                        pred = PREDICTION_PATTERN.match(f.readline()).group(1).strip()
                        truth = GROUND_TRUTH_PATTERN.match(f.readline()).group(1).strip()

                        if se == sentence and dfa_equality == 0:
                            result.append((file, pred))

                    line = f.readline()
    pprint(result)
    print(len(result))
    return result


def read_analysis_file():
    analysis_list = list()
    with open(ANALYSIS_PATH, "r") as f:
        for line in f:
            analysis_list.append(line.strip())
    return analysis_list


def batch(path):

    sentences = read_analysis_file()

    for sentence in sentences:
        result = calc_err(path, sentence)

        with open("result.log", "w") as f:
            f.write("=============================================\n")
            f.write(sentence)
            f.write('\n'.join([', '.join(r) for r in result]))
            f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate DFA Equality')
    parser.add_argument('--path', help='epoch file', required=True)
    parser.add_argument('--sentence', help='sentence')
    parser.add_argument('--format', help='file format (text or json)')

    args = parser.parse_args()

    if args.sentence and args.format:
        calc_err(args.path, args.sentence, args.format)
    else:
        batch(args.path)

