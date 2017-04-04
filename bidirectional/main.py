# coding=utf8

import os
import time
import json
import argparse
import subprocess
import numpy as np
import tensorflow as tf

from model import Model
from data_provider.data_iterator import VocabManager, DataIterator
from tools.test_dfa_equality import process as dfa_process

JAR_PATH = os.path.join(os.getcwd(), 'regex_dfa_equals.jar')


# np.set_printoptions(threshold=np.nan)


def read_configuration(path):
    with open(path, "r") as f:
        return json.load(f)


class ModelRuntime:
    def __init__(self, configuration):

        self._conf = read_configuration(configuration)

        self._vocab_manager = VocabManager(self._conf["vocab_files"])

        self._epoches = self._conf["epoches"]
        self._batch_size = self._conf["batch_size"]
        self._max_length_encode = self._conf["max_length_encode"]
        self._max_length_decode = self._conf["max_length_decode"]
        self._log_dir = self._conf["log_dir"]
        self._result_log_base_path = os.path.abspath(self._conf["result_log"] + str(int(time.time())))
        self._is_test_capability = self._conf["is_test_capability"]

        os.mkdir(self._result_log_base_path)
        self._save_conf_file()

        self._train_data_iterator = DataIterator(
            self._conf["train_file"],
            self._max_length_encode,
            self._max_length_decode
        )

        self._test_data_iterator = DataIterator(
            self._conf["test_file"],
            self._max_length_encode,
            self._max_length_decode
        )

        self._development_data_iterator = DataIterator(
            self._conf["development_file"],
            self._max_length_encode,
            self._max_length_decode
        )

        self._session = None
        self._summary_operator = None
        self._file_writer = None
        self._test_model = None
        self._train_model = None

        self._cached_logs = dict()

    def _save_conf_file(self):
        """
        Save Configuration to result log directory
        :return:
        """
        path = os.path.join(self._result_log_base_path, "config.json")
        with open(path, "w") as f:
            f.write(json.dumps(self._conf, indent=4))

    def init_session(self):
        self._session = tf.Session()

        with tf.variable_scope("seq2seq") as scope:
            self._train_model = Model(self._vocab_manager, self._conf, is_test=False)
            scope.reuse_variables()
            self._test_model = Model(self._vocab_manager, self._conf, is_test=True)
            init = tf.global_variables_initializer()
            self._session.run(init)
            self._file_writer = tf.summary.FileWriter(self._log_dir, self._session.graph)
            self._summary_operator = tf.summary.merge_all()

    def _write(self, file=None):
        """
        Force to write to file
        :return:
        """
        if file:
            with open(file, "a") as f:
                for log in self._cached_logs[file]:
                    f.write(log)
            self._cached_logs[file] = list()
            return

        for log_file, value in self._cached_logs.items():
            with open(log_file, "a") as f:
                for log in value:
                    f.write(log)
            self._cached_logs[log_file] = list()

    def _log_test(self, file, source, ground_truth, prediction, diff, dfa_diff):

        source = 'S: ' + self._vocab_manager.decode_source(source)
        padded_seq_str = 'p: ' + self._vocab_manager.decode(prediction)
        ground_truth_str = 'T: ' + self._vocab_manager.decode(ground_truth)
        segmentation = '============================================='
        summary = 'exact_match_diff: %d' % diff
        dna_equality = 'dfa_equality: %d' % dfa_diff
        string = '\n'.join(['\n', summary, dna_equality, source, padded_seq_str, ground_truth_str, segmentation])

        if file not in self._cached_logs:
            self._cached_logs[file] = []
        self._cached_logs[file].append(string)

        if len(self._cached_logs[file]) >= 100:
            self._write(file)

    def _log_epoch(self, epoch, accuracy, dfa_accuracy, train_accuracy, loss):
        result_log = os.path.join(self._result_log_base_path, "result_log.txt")
        with open(result_log, "a") as f:
            log = "\n epoch_num: %f, accuracy: %f, dfa_accuracy: %f, train_accuracy: %f, average_loss: %f \n" % (
                epoch, accuracy, dfa_accuracy, train_accuracy, loss)
            f.write(log)

    def _calc_accuracy(self, ground_truth, prediction, is_dfa_test=True):
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

        gold = _replace(self._vocab_manager.decode(ground_truth))
        pred = _replace(self._vocab_manager.decode(prediction))
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

    def test(self, log_file=None):
        # TODO Batch test
        """
        Test Model
        :return:
        """

        def _test():
            sample = self._test_data_iterator.get_batch(1)
            last_predictions, _predictions, logprobs, mask, decoder_states, feed_dict = self._test_model.predict(
                sample.encoder_seq, sample.source_length)
            last_predictions, _predictions, logprobs, mask, decoder_states = self._session.run(
                (last_predictions, _predictions, logprobs, mask, decoder_states), feed_dict=feed_dict)

            if np.sum(mask) == 0:
                index = np.argmax(logprobs)
            else:
                index = np.argmin(logprobs * (-mask))
            _predictions = _predictions[index]
            ground_truth = np.array(sample.target_seq[0])
            padded_seq = np.concatenate((_predictions, np.array([0] * (len(ground_truth) - len(_predictions)))), axis=0)
            exact_match, dfa_equality = self._calc_accuracy(ground_truth, padded_seq, is_dfa_test=False)

            if log_file:
                self._log_test(log_file, sample.encoder_seq[0], ground_truth, padded_seq, exact_match, dfa_equality)
            return 1, exact_match, dfa_equality

        total = 0
        exact_correct = 0
        dfa_corrent = 0
        for i in range(self._test_data_iterator.size):
            t, e, d = _test()
            total += t
            exact_correct += e
            dfa_corrent += d
        return exact_correct / total, dfa_corrent / total

    def test_on_development_set(self):
        # TODO Batch test
        """
        Test on development set
        :return:
        """

        def _test():
            sample = self._development_data_iterator.get_batch(1)
            last_predictions, _predictions, logprobs, mask, decoder_states, feed_dict = self._test_model.predict(
                sample.encoder_seq, sample.source_length)
            last_predictions, _predictions, logprobs, mask, decoder_states = self._session.run(
                (last_predictions, _predictions, logprobs, mask, decoder_states), feed_dict=feed_dict)

            if np.sum(mask) == 0:
                index = np.argmax(logprobs)
            else:
                index = np.argmin(logprobs * (-mask))
            _predictions = _predictions[index]
            ground_truth = np.array(sample.target_seq[0])
            padded_seq = np.concatenate((_predictions, np.array([0] * (len(ground_truth) - len(_predictions)))), axis=0)
            exact_match, dfa_equality = self._calc_accuracy(ground_truth, padded_seq, is_dfa_test=False)
            return 1, exact_match, dfa_equality

        total = 0
        exact_correct = 0
        dfa_corrent = 0
        for i in range(self._development_data_iterator.size):
            t, e, d = _test()
            total += t
            exact_correct += e
            dfa_corrent += d
        return exact_correct / total, dfa_corrent / total

    def _calc_train_set_accuracy(self, predictions, ground_truths):
        """
        Check accuracy in training set
        :return:
        """
        exact_correct = 0
        dfa_corrent = 0
        for (prediction, ground_truth) in zip(predictions, ground_truths):
            ground_truth = np.array(ground_truth)
            padded_seq = np.concatenate((prediction, np.array([0] * (len(ground_truth) - len(prediction)))), axis=0)
            e, d = self._calc_accuracy(ground_truth, padded_seq, is_dfa_test=False)
            exact_correct += e
            dfa_corrent += d
        return exact_correct, dfa_corrent

    def train(self):
        """
        Train model
        :return:
        """
        losses = list()

        def evaluate(epoch_num):
            nonlocal i
            nonlocal losses
            nonlocal train_exact_match
            nonlocal total
            i += 1
            loss = np.average(np.array(losses))
            print(losses)

            if self._is_test_capability:
                train_accuracy = train_exact_match / total
                string = "epoch_num: %f, train_accuracy: %f, average_loss: %f" % (epoch_num, train_accuracy, loss)
                print(string)
                with open(os.path.join(self._result_log_base_path, "result.txt"), "a") as f:
                    f.write(string + "\n")
            else:
                log_file = os.path.join(self._result_log_base_path, "epoch_%d.txt" % epoch_num)
                accuracy, dfa_accuracy = self.test(log_file)
                development_accuracy, development_dfa_accuracy = self.test_on_development_set()
                train_accuracy = train_exact_match / total
                print(
                    "epoch_num: %f, train_accuracy: %f, development_accuracy: %f, test_accuracy: %f, average_loss: %f " % (
                        epoch_num, train_accuracy, development_accuracy, accuracy, loss))
                self._log_epoch(epoch_num, accuracy, dfa_accuracy, train_accuracy, loss)
            losses = list()
            total = 0
            train_exact_match = 0

        self._train_data_iterator.epoch_cb = evaluate

        i = 0
        train_exact_match = 0
        total = 0
        while i < self._epoches:
            batch = self._train_data_iterator.get_batch(self._batch_size)
            prediction, loss, optimizer, gvs, clipped_gvs, feed = self._train_model.train(batch)
            prediction, loss, gvs, clipped_gvs, optimizer = self._session.run(
                (prediction, loss, gvs, clipped_gvs, optimizer,),
                feed_dict=feed)
            exact_match, dfa_correct = self._calc_train_set_accuracy(prediction, batch.target_seq)
            train_exact_match += exact_match
            total += batch.batch_size
            losses.append(loss)

        # Force Write
        self._write()

        # Evaluate
        for j in range(i):
            filename = "epoch_%d.txt" % (j+1)
            dfa_process(os.path.join(self._result_log_base_path, filename))

    def run(self):
        self.train()
        self._session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--conf", help="Configuration File")
    args = parser.parse_args()

    runtime = ModelRuntime(args.conf)
    runtime.init_session()
    runtime.run()
