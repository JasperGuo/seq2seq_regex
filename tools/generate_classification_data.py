# coding=utf8

"""
Generate Data for classification:
    Given two regular expression, predict whether they are functionality equal or not!
"""

import os
import re
import argparse
from multiprocessing import Queue, Process, Lock

TARGET_FILE_PATH = "regex_classification_dataset.txt"

STRING_MATCH_PATTERN = re.compile(r'exact_match_diff: (\d)\n')
DFA_PATTERN = re.compile(r'dfa_equality: (\d)\n')
SENTENCE_PATTERN = re.compile(r'S: (.*)\n')
PREDICTION_PATTERN = re.compile(r'p: (.*)\n')
GROUND_TRUTH_PATTERN = re.compile(r'T: (.*)\n')

CACHE_LOGS = 400
WORKER_NUM = 2

STRING_MATCH = "STRING_MATCH"
DFA_MATCH = "DFA_MATCH"

DFA_MATCH_TARGET_FILE_PATH = "dfa_match_dataset.txt"
STRING_MATCH_FILE_PATH = "string_match_dataset.txt"


def process_worker(file_queue, result_queue, lock):
    """
    Process file, find regex pairs
    :param file_queue:
    :param result_queue:
    :return:
    """
    while True:
        file = file_queue.get()
        if not file:
            break
        else:
            with open(file, "r") as f:
                line = f.readline()
                while line and line != "":
                    match = STRING_MATCH_PATTERN.match(line)
                    if match:
                        score = int(match.group(1).strip())
                        dfa_equality_match = int(DFA_PATTERN.match(f.readline()).group(1).strip())
                        sentences_match = SENTENCE_PATTERN.match(f.readline())
                        regex_1 = PREDICTION_PATTERN.match(f.readline()).group(1).strip()
                        regex_2 = GROUND_TRUTH_PATTERN.match(f.readline()).group(1).strip()
                        score = (regex_1 == regex_2)
                        if dfa_equality_match == 1 and score == 0:
                            msg = {
                                "regex_1": regex_1,
                                "regex_2": regex_2,
                                "tag": DFA_MATCH
                            }
                            result_queue.put(msg)
                        elif dfa_equality_match == 1 and score == 1:
                            msg = {
                                "regex_1": regex_1,
                                "regex_2": regex_2,
                                "tag": STRING_MATCH
                            }
                            # queue.put(msg)
                    line = f.readline()


def _write(tag, logs):
    if tag == DFA_MATCH:
        filename = DFA_MATCH_TARGET_FILE_PATH
    elif tag == STRING_MATCH:
        filename = STRING_MATCH_FILE_PATH
    else:
        return

    with open(filename, "a") as f:
        for log in logs:
            string = '\t'.join([log["regex_1"], log["regex_2"]])
            f.write(string + "\n")


def write_worker(queue):
    cached_logs = {
        DFA_MATCH: [],
        STRING_MATCH: []
    }
    while True:
        msg = queue.get()
        if not msg:
            for tag, logs in cached_logs.items():
                _write(tag, logs)
                cached_logs[tag] = list()
            break
        else:
            cached_logs[msg["tag"]].append(msg)

            if len(cached_logs[msg["tag"]]) >= CACHE_LOGS:
                _write(msg["tag"], cached_logs[msg["tag"]])
                cached_logs[msg["tag"]] = list()


def read(path, file_queue):
    for filename in os.listdir(path):
        full_path = os.path.join(path, filename)
        if os.path.isdir(full_path):
            read(full_path, file_queue)
        else:
            file_queue.put(full_path)


def main(path):
    processors = list()
    lock = Lock()
    file_queue = Queue()
    result_queue = Queue()

    for i in range(WORKER_NUM):
        worker_p = Process(target=process_worker, args=(file_queue, result_queue, lock))
        worker_p.start()
        processors.append(worker_p)

    result_worker = Process(target=write_worker, args=(result_queue,))
    result_worker.start()

    read(path, file_queue)

    for i in range(WORKER_NUM):
        file_queue.put(None)

    for p in processors:
        p.join()

    result_queue.put(None)
    result_worker.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate DFA Equality')
    parser.add_argument('--file', help='epoch file', required=True)

    args = parser.parse_args()
    main(args.file)