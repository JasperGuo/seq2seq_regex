# coding=utf8

"""
Refine Regex Expression test case, shorten the random string
"""
import argparse
from multiprocessing import Process, Queue
from util import save_data, read_data
import re
import random

MAX_CHANCE = 20


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


def _replace_case(case):
    case = case.replace("<M0>", 'dog')
    case = case.replace("<M1>", 'truck')
    case = case.replace("<M2>", 'ring')
    case = case.replace("<M3>", 'lake')
    return case

MATCH = "MATCH"
NOT_MATCH = "NOT_MATCH"


def worker(regex_pattern, string, queue):
    match = regex_pattern.findall(string)
    if match:
        queue.put(MATCH)
    else:
        queue.put(NOT_MATCH)


def shorten_test_case(file_path, target_path, error_path, max_len, timeout):
    """
    shorten the length of each test case
    :param file_path:
    :param target_path:
    :param error_path:
    :param max_len:
    :param timeout:
    :return:
    """
    original_data = read_data(file_path)
    processed_data = list()
    fail_list = list()
    for sample in original_data:
        case = sample["case"]
        if len(case) <= max_len:
            processed_data.append(sample)
        else:
            length = len(case)
            parts = round(length/max_len)
            pattern = re.compile(_replace(sample["regex"]))
            for i in range(parts):
                string = case[i*max_len: (i+1)*max_len]

                _process = Process(target=worker, args=(pattern, string,))
                _process.start()
                _process.join(timeout=timeout)
                _process.terminate()

                if _process.exitcode == MATCH and sample["label"] == 1:
                    sample["case"] = string
                    processed_data.append(sample)
                    break
                elif (_process.exitcode == NOT_MATCH) and sample["label"] == 0:
                    sample["case"] = string
                    processed_data.append(sample)
                    break
            else:
                # Fail to shorten the case
                fail_list.append(sample)
                print(sample["regex"], case)

    save_data(target_path, processed_data)
    save_data(error_path, fail_list)


def larger_error_case(file_path, target_path, timeout):
    """
    Construct higher quality negative test case
    :param file_path:
    :param target_path:
    :param timeout:
    :return:
    """
    queue = Queue()
    original_data = read_data(file_path)
    case_dict = dict()
    for sample in original_data:
        sentence = sample["sentence"]
        if sentence not in case_dict:
            case_dict[sentence] = {
                "positive": [],
                "negative": [],
                "regex": sample["regex"]
            }
        label = sample["label"]
        if label == 1:
            case_dict[sentence]["positive"].append(sample["case"])
        else:
            case_dict[sentence]["negative"].append(sample["case"])

    sentences = list(case_dict.keys())
    for sentence, values in case_dict.items():
        print(sentence)
        regex = re.compile(_replace(values["regex"]))
        negative_cases = values["negative"]
        i, j = 0, 0
        num_cases = len(negative_cases)
        while i < MAX_CHANCE and j < num_cases:
            # Sample a different Sentence
            sample_sentence = random.sample(sentences, 1)[0]
            sample_positive_cases = case_dict[sample_sentence]["positive"]
            while sample_sentence == sentence or len(sample_positive_cases) == 0:
                sample_sentence = random.sample(sentences, 1)[0]
                sample_positive_cases = case_dict[sample_sentence]["positive"]

            sample_case = random.sample(sample_positive_cases, 1)[0]
            curr_negative_case = negative_cases[j]

            new_case = sample_case + curr_negative_case

            _process = Process(target=worker, args=(regex, _replace_case(new_case), queue))
            _process.start()
            _process.join(timeout=timeout)
            if _process.is_alive():
                result = None
            else:
                result = queue.get()
            _process.terminate()
            if result == NOT_MATCH:
                values["negative"][j] = new_case
                j += 1
            i += 1

    result = list()
    for sentence, values in case_dict.items():
        regex = values["regex"]
        for case in values["positive"]:
            result.append({
                "sentence": sentence,
                "regex": regex,
                "case": case,
                "label": 1
            })
        for case in values["negative"]:
            result.append({
                "sentence": sentence,
                "regex": regex,
                "case": case,
                "label": 0
            })
    save_data(target_path, result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", help="source file")
    parser.add_argument("--target", help="target file")
    # parser.add_argument("--error", help="error file")
    # parser.add_argument("--length", help="case max length")
    parser.add_argument("--timeout", help="timeout")

    args = parser.parse_args()

    # shorten_test_case(args.source, args.target, args.error, int(args.length), int(args.timeout))
    larger_error_case(args.source, args.target, int(args.timeout))

