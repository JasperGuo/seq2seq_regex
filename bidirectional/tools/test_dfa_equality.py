# coding=utf8

import argparse
import re
import os
from multiprocessing import Process, Queue

import subprocess

STRING_MATCH_PATTERN = re.compile(r'exact_match_diff: (\d)\n')
DFA_PATTERN = re.compile(r'dfa_equality: (\d)\n')
SENTENCE_PATTERN = re.compile(r'S: (.*)\n')
PREDICTION_PATTERN = re.compile(r'p: (.*)\n')
GROUND_TRUTH_PATTERN = re.compile(r'T: (.*)\n')
SCORE_PATTERN = re.compile(r'score: ([-+]?([0-9]*\.[0-9]+|[0-9]+))')

PARALLEL_EVALUATION = 10
CACHE_LOGS = 100

WORKER_NUM = 8


def _calc_accuracy(ground_truth, prediction, is_dfa_test=True):
    """
    Calculate the accuracy
    :param ground_truth:
    :param prediction:
    :return: (boolean, boolean)
        String-Equality
        DFA-Equality
    """
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

    gold = _replace(ground_truth.strip())
    pred = _replace(prediction.strip())
    diff = (gold == pred)

    if not is_dfa_test:
        return diff, False
    try:
        out = subprocess.check_output(['java', '-jar', 'regex_dfa_equals.jar', gold, pred])
        dfa_result = '\n1' in out.decode()
    except Exception as e:
        print(e)
        dfa_result = False
    return diff, dfa_result


def worker(work_queue, result_queue):
    """
    Check DFA Equality
    :param work_queue:
    :param result_queue:
    :return:
    """
    while True:
        msg = work_queue.get()
        if msg == 'Done':
            break
        else:
            diff, dfa_result = _calc_accuracy(msg["truth"], msg["prediction"])
            msg["dfa_equality"] = dfa_result
            result_queue.put(msg)


def write_worker(result_queue, result_path):
    """
    Write logs to file
    :param result_path:
    :param result_queue:
    :return:
    """

    cached_logs = dict()
    curr_len = 0

    def _write(include_statistic=False):
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        for file_name, value in cached_logs.items():
            with open(os.path.join(result_path, file_name), "a") as f:
                for log in value["logs"]:
                    f.write(log)
                if include_statistic:
                    f.write("\n" + "accuracy: %f, dfa_accuracy: %f" % (value["correct"]/value["total"], value["dfa_correct"]/value["total"]))
                cached_logs[file_name]["logs"] = []
    while True:
        msg = result_queue.get()
        if msg == "Done":
            _write(include_statistic=True)
            break
        else:
            source = 'S: ' + msg["sentence"]
            padded_seq_str = 'p: ' + msg["prediction"]
            ground_truth_str = 'T: ' + msg["truth"]
            segmentation = '============================================='
            summary = 'exact_match_diff: %d' % msg["score"]
            dfa_equality = 'dfa_equality: %d' % msg["dfa_equality"]
            logprob = 'score: %f' % msg["logprob"]
            string = '\n'.join(['\n', summary, dfa_equality, logprob, source, padded_seq_str, ground_truth_str, segmentation])

            curr_len += 1

            file_name = msg["file_name"]

            if file_name not in cached_logs:
                cached_logs[file_name] = {
                    "logs": [],
                    "correct": 0,
                    "dfa_correct": 0,
                    "total": 0
                }
            cached_logs[file_name]["logs"].append(string)
            cached_logs[file_name]["correct"] += msg["score"]
            cached_logs[file_name]["dfa_correct"] += msg["dfa_equality"]
            cached_logs[file_name]["total"] += 1

            if curr_len >= CACHE_LOGS:
                _write()
                curr_len = 0


def read(file_path):
    result = list()
    file_name = os.path.basename(file_path)
    with open(file_path, "r") as f:
        line = f.readline()
        while line and line != "":
            match = STRING_MATCH_PATTERN.match(line)
            if match:
                score = int(match.group(1).strip())
                dfa_equality_match = DFA_PATTERN.match(f.readline())
                logprob = float(SCORE_PATTERN.match(f.readline()).group(1).strip())
                sentences_match = SENTENCE_PATTERN.match(f.readline())
                result.append({
                    "file_name": '_'.join(["dfa", file_name]),
                    "dfa_equality": dfa_equality_match.group(1).strip(),
                    "score": score,
                    "logprob": logprob,
                    "sentence": sentences_match.group(1).strip(),
                    "prediction": PREDICTION_PATTERN.match(f.readline()).group(1).strip(),
                    "truth": GROUND_TRUTH_PATTERN.match(f.readline()).group(1).strip()
                })
            line = f.readline()
    return result


def process(file_path):
    predict_results = read(file_path)

    result_path = os.path.join(os.path.dirname(file_path), "dfa_evaluation")

    worker_queue = Queue()
    result_queue = Queue()

    processes = list()
    for i in range(WORKER_NUM):
        worker_p = Process(target=worker, args=(worker_queue, result_queue,))
        worker_p.start()
        processes.append(worker_p)

    result_worker = Process(target=write_worker, args=(result_queue, result_path))
    result_worker.start()

    for predict_result in predict_results:
        worker_queue.put(predict_result)

    for i in range(WORKER_NUM):
        worker_queue.put("Done")

    for p in processes:
        p.join()

    result_queue.put("Done")
    result_worker.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate DFA Equality')
    parser.add_argument('--file', help='epoch file', required=True)

    args = parser.parse_args()
    process(args.file)


