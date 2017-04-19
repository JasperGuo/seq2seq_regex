# coding=utf8
import os
import re
import argparse
import random
import numpy as np
from multiprocessing import Process, Queue
from util import read_data, save_data


MATCH = "MATCH"
NOT_MATCH = "NOT_MATCH"
MAX_CHANCE = 20


def _replace_regex(regex):
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


def _replace_case_reverse(case):
    _case = case.replace("dog", '<M0>')
    _case = _case.replace("truck", '<M1>')
    _case = _case.replace("ring", '<M2>')
    _case = _case.replace("lake", '<M3>')
    return _case


def flatten_data(file_path):
    raw_data = read_data(file_path)

    data = list()

    for sample in raw_data:
        sentence = sample["sentence"]
        regex = sample["regex"]
        positive_cases = sample["cases"]["positive"]
        negative_cases = sample["cases"]["negative"]
        for case in positive_cases:
            data.append({
                "sentence": sentence,
                "case": case,
                "label": 1,
                "regex": regex
            })
        for case in negative_cases:
            data.append({
                "sentence": sentence,
                "case": case,
                "label": 0,
                "regex": regex
            })
    return data


def worker(regex_pattern, string, queue):
    match = regex_pattern.findall(string)
    if match:
        queue.put(MATCH)
    else:
        queue.put(NOT_MATCH)


def shorten_test_case(data, max_len, timeout):
    """
    shorten the length of each test case
    :param data
    :param max_len:
    :param timeout:
    :return:
    """
    original_data = data
    queue = Queue()
    processed_data = list()
    fail_list = list()
    for sample in original_data:
        case = sample["case"]
        if len(case) <= max_len:
            processed_data.append(sample)
        else:
            length = len(case)
            parts = round(length/max_len)
            pattern = re.compile(_replace_regex(sample["regex"]))
            for i in range(parts):
                string = case[i*max_len: (i+1)*max_len]

                _process = Process(target=worker, args=(pattern, string, queue))
                _process.start()
                _process.join(timeout=timeout)
                if _process.is_alive():
                    result = None
                else:
                    result = queue.get()
                _process.terminate()

                if result == MATCH and sample["label"] == 1:
                    sample["case"] = string
                    processed_data.append(sample)
                    break
                elif (result == NOT_MATCH) and sample["label"] == 0:
                    sample["case"] = string
                    processed_data.append(sample)
                    break
            else:
                # Fail to shorten the case
                fail_list.append(sample)
                print(sample["regex"], case)
    return processed_data, fail_list


def larger_error_case(data, timeout):
    """
    Construct higher quality negative test case
    :param timeout:
    :param data
    :return:
    """
    queue = Queue()
    original_data = data
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
        regex = re.compile(_replace_regex(values["regex"]))
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
    return result


def replace_key_words(data):

    for sample in data:
        sample["case"] = _replace_case_reverse(sample["case"])

    return data


def split(data):
    raw_data = data

    for sample in raw_data:
        case = sample["case"]
        case = case.replace(" ", "")
        if "<M0>" in case:
            case = case.replace("<M0>", " <M0> ")
        if "<M1>" in case:
            case = case.replace("<M1>", " <M1> ")
        if "<M2>" in case:
            case = case.replace("<M2>", " <M2> ")
        if "<M3>" in case:
            case = case.replace("<M3>", " <M3> ")

        i = 0
        length = len(case)
        new_string = ""
        while i < length:
            if str.isspace(case[i]):
                j = i+1
                while (not str.isspace(case[j])) and j < length:
                    j += 1
                word = case[i+1:j]
                i = j + 1
            else:
                word = case[i]
                i += 1
            new_string += " " + word
        sample["case"] = new_string.strip()
    return raw_data


def remove_empty_case(raw_data):
    result = list()
    for sample in raw_data:
        case = sample["case"]
        if case and len(case.replace(" ", "")) == 0:
            continue
        result.append(sample)
    return result


def main(file_path, target_path, fail_path, timeout, max_length):
    flattened_data = flatten_data(file_path)
    shortened_positive_case_data, fail_data = shorten_test_case(flattened_data, max_length, timeout)
    larger_negative_case_data = larger_error_case(shortened_positive_case_data, timeout)
    replaced_key_words = replace_key_words(larger_negative_case_data)
    split_data = split(replaced_key_words)
    processed_data = remove_empty_case(split_data)

    save_data(target_path, processed_data)
    save_data(fail_path, fail_data)

    print("Statistic Analysis: ")
    negative_length, positive_length = list(), list()
    for sample in processed_data:
        label = sample["label"]
        case = sample["case"]
        if label == 0:
            negative_length.append(len(case))
        else:
            positive_length.append(len(case))
    print("Positive Case: %d, Negative Case: %d" % (len(positive_length), len(negative_length)))
    print("Positive Case Average Length: %f" % np.average(positive_length))
    print("Negative Case Average Length: %f" % np.average(negative_length))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", help="source file")
    parser.add_argument("--target", help="target file")
    parser.add_argument("--error", help="error file")
    parser.add_argument("--length", help="case max length")
    parser.add_argument("--timeout", help="timeout")
    args = parser.parse_args()

    main(args.source, args.target, args.error, int(args.timeout), int(args.length))
