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


def main(baseline, intra, bilstm, search):
    """
    :param baseline:
    :param intra:
    :param bilstm:
    :param search:
    :return:
    """
    intra_dict = _build_set(intra)
    bilstm_dict = _build_set(bilstm)
    search_dict = _build_set(search)
    baseline_dict = _build_set(baseline)

    result_path = "error_compare_log.txt"

    for sentence, regex_dict in baseline_dict.items():
        logs = []
        if sentence not in intra_dict:
            logs.append("intra")
        if sentence not in bilstm_dict:
            logs.append("bilstm")
        if sentence not in search_dict:
            logs.append("beam_search")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate DFA Equality')
    parser.add_argument('--baseline', help='baseline file', required=True)
    parser.add_argument('--intra', help='intra attention file', required=True)
    parser.add_argument('--bilstm', help='beam search file', required=True)
    parser.add_argument('--search', help='bidirectional lstm file', required=True)

    args = parser.parse_args()
    main(args.baseline, args.intra, args.bilstm, args.search)