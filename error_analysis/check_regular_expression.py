# coding=utf8

"""
Check Regular expression
"""

import re
import json
import argparse


def find_test_case(sentence):
    with open("dataset\\test.json", "r") as f:
        data_set = json.load(f)

    for case in data_set:
        if case["sentence"] == sentence:
            return case

    return None


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


def process(sentence, test_regex):
    test_case = find_test_case(sentence)
    if not test_case:
        print("None test case found")
        return

    ground_truth = test_case["regex"].replace(" ", "")
    positive_case = test_case["cases"]["positive"]
    negative_case = test_case["cases"]["negative"]

    predicted_regex = re.compile(_replace(test_regex))
    truth_regex =re.compile(_replace(ground_truth))

    print("============== Positive ==================")
    for idx, case in enumerate(positive_case):
        match = truth_regex.match(case)
        test_match = predicted_regex.match(case)
        print(idx, "truth_match: %d, test_match: %d" % (1 if match else 0, 1 if test_match else 0))

    print("============== Negative ==================")
    for idx, case in enumerate(negative_case):
        match = truth_regex.match(case)
        test_match = predicted_regex.match(case)
        print(idx, "truth_match: %d, test_match: %d" % (1 if match else 0, 1 if test_match else 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate DFA Equality')
    parser.add_argument('--sentence', help='sentence', required=True)
    parser.add_argument('--regex', help='regular expression', required=True)

    args = parser.parse_args()
    process(args.sentence, args.regex)
