# coding=utf8

"""
Evaluate the syntax of the generated regular expressions
"""

import argparse
import re

PATTERN = re.compile(r'p:(.*)\n')


def read(file_path):
    result = list()
    with open(file_path, "r") as f:
        for line in f:
            matches = PATTERN.match(line)
            if matches:
                regex = matches.group(1).strip()
                result.append(regex)
    return result


def check(regex):
    """
    Check Syntax, focus on the bracket
    :param regex:
    :return:
    """
    push_bracket = ['(', '[', '{']
    bracket_matches = {
        "}": "{",
        "]": "[",
        ")": "("
    }
    stack = list()
    for char in regex:
        if char in push_bracket:
            stack.append(char)
        elif char in bracket_matches.keys():
            corresponding_char = bracket_matches[char]
            if len(stack) == 0 or stack[-1] != corresponding_char:
                break
            else:
                stack.pop()
    else:
        if len(stack) == 0:
            return True
    return False


def dump_error(error_list):
    with open("error_list.txt", "a") as f:
        errors = '\n'.join(error_list)
        segmentation = "======== Error List ==================================="
        f.write('\n'.join([segmentation, errors, '\n']))


def main(file_path):
    result = read(file_path)
    total = len(result)

    assert total > 0

    correct = 0
    error_list = list()
    for regex in result:
        if check(regex):
            correct += 1
        else:
            error_list.append(regex)
    print(error_list)
    print("Accuracy: %f" % (correct/total))
    dump_error(error_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Regular Expression Syntax Checker')
    parser.add_argument('--file', help='epoch file', required=True)

    args = parser.parse_args()
    main(args.file)
