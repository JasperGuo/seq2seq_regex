# coding=utf8

import argparse
import json


def _build_set(file):
    with open(file, "r") as f:
        result = json.load(f)
    sentence_dict = dict()
    for error in result:
        sentence = error["sentence"]
        del error["sentence"]
        sentence_dict[sentence] = error
    return sentence_dict


def main(baseline, bilstm):
    """
    :param baseline:
    :param bilstm:
    :return:
    """
    bilstm_dict = _build_set(bilstm)
    baseline_dict = _build_set(baseline)

    result_path = "error_compare_log.txt"
    solved_path = "solved_log.txt"
    unsolved_path = "unsolved_log.txt"

    for sentence, regex_dict in baseline_dict.items():
        logs = []
        is_solved = False

        if sentence not in bilstm_dict:
            logs.append("bilstm")
            is_solved = True

        if len(logs) == 0:
            log = "Not Solved"
        else:
            log = ', '.join(["Solved by ", ', '.join(logs)])

        with open(result_path, "a") as f:
            f.write("===============================================\n")
            f.write(sentence)
            f.write("\n")
            f.write(log)
            f.write("\n")
        if is_solved:
            with open(solved_path, "a") as f:
                f.write("===============================================\n")
                f.write(sentence)
                f.write("\n")
                f.write(log)
                f.write("\n")
        else:
            with open(unsolved_path, "a") as f:
                f.write("===============================================\n")
                f.write(sentence)
                f.write("\n")
                f.write(log)
                f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate DFA Equality')
    parser.add_argument('--baseline', help='baseline file', required=True)
    parser.add_argument('--bilstm', help='beam search file', required=True)

    args = parser.parse_args()
    main(args.baseline, args.bilstm)