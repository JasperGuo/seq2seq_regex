# coding=utf8

"""
1. Group positive test case with sentence
2. Check Positive Case number per sentence
"""

import argparse
from util import read_data, save_data


def group_data(raw_data):
    sentence_dict = dict()
    for sample in raw_data:
        label = sample["label"]
        if label == 0:
            continue
        sentence = sample["sentence"]
        if sentence not in sentence_dict:
            sentence_dict[sentence] = {
                "positive_case": [],
                "regex": sample["regex"]
            }
        sentence_dict[sentence]["positive_case"].append(sample["case"])
    return sentence_dict


def check_sentence_positive_num(sen_case):
    remove_list = list()
    for sample in sen_case:
        if len(sample["positive_case"]) < 5:
            remove_list.append(sample)
    for r in remove_list:
        sen_case.remove(r)
    return sen_case


def reformat(sentence_dict):
    result = list()
    for sentence, values in sentence_dict.items():
        r = {
            "sentence": sentence
        }
        r.update(values)
        result.append(r)
    return result


def main(filepath, target_path):
    data = read_data(filepath)
    sentence_dict = group_data(data)
    result = reformat(sentence_dict)
    result = check_sentence_positive_num(result)
    print(len(result))
    save_data(target_path, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", help="source file")
    parser.add_argument("--target", help="target file")
    args = parser.parse_args()

    main(args.source, args.target)
