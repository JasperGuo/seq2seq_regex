# coding=utf8

# coding=utf8

"""
Analyze the reason why accuracy is low
"""

import argparse
import re
import json
from pprint import pprint

PATTERN = re.compile(r'dfa_equality: (\d)\n')
SENTENCE_PATTERN = re.compile(r'S: (.*)\n')
PREDICTION_PATTERN = re.compile(r'p: (.*)\n')
GROUND_TRUTH_PATTERN = re.compile(r'T: (.*)\n')


def read(file_path):
    result = list()
    with open(file_path, "r") as f:
        line = f.readline()
        while line and line != "":
            match = PATTERN.match(line)
            if match:
                score = int(match.group(1).strip())
                sentences_match = SENTENCE_PATTERN.match(f.readline())
                result.append({
                    "score": score,
                    "sentence": sentences_match.group(1).strip(),
                    "prediction": PREDICTION_PATTERN.match(f.readline()).group(1).strip(),
                    "truth": GROUND_TRUTH_PATTERN.match(f.readline()).group(1).strip()
                })
            line = f.readline()
    return result


def save(path, content):
    with open(path, "a") as f:
        f.write(json.dumps(content, indent=4))
        f.write("\n===========================================\n")


def main(file_path):
    generate_results = read(file_path)

    conjunction = ['and', 'or']

    total = len(generate_results)
    conjunction_sentence = 0
    correct = 0
    correct_in_conjunction = 0

    without_conjunctions = list()

    for generate_result in generate_results:
        correct += generate_result["score"]
        for conj in conjunction:
            if conj in generate_result["sentence"] or conj.upper() in generate_result["sentence"]:
                conjunction_sentence += 1
                correct_in_conjunction += generate_result["score"]
                break
        else:
            without_conjunctions.append(generate_result)

    total_accuracy = correct/total
    accuracy_in_conjunction = correct_in_conjunction/conjunction_sentence
    accuracy_in_non_conjunction = (correct - correct_in_conjunction)/(total - conjunction_sentence)

    print("Total accuracy: %f" % total_accuracy)
    print("Correct in conjunction sentence: %f" % accuracy_in_conjunction)
    print("Correct in non conjunction sentence: %f" % accuracy_in_non_conjunction)
    print("Total sentence: %d" % total)
    print("Total conjunction: %d" % conjunction_sentence)

    save("analyze_without_conjunction.txt", without_conjunctions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Regular Expression Syntax Checker')
    parser.add_argument('--file', help='epoch file', required=True)

    args = parser.parse_args()
    main(args.file)
