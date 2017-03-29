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


JAR_PATH = os.path.join(os.getcwd(), 'regex_dfa_equals.jar')


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

        self._result_log_base_path = self._conf["result_log"] + str(int(time.time()))
        os.mkdir(self._result_log_base_path)
        self._result_log = None

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

    def _log_test(self, source, ground_truth, prediction, diff, dfa_diff):
        with open(self._result_log, "a") as f:
            source = 'S: ' + self._vocab_manager.decode_source(source)
            padded_seq_str = 'p: ' + self._vocab_manager.decode(prediction)
            ground_truth_str = 'T: ' + self._vocab_manager.decode(ground_truth)
            segmentation = '============================================='
            summary = 'exact_match_diff: %d' % diff
            dna_equality = 'dfa_equality: %d' % dfa_diff
            f.write('\n'.join(['\n', summary, dna_equality, source, padded_seq_str, ground_truth_str, segmentation]))

    def _log_epoch(self, epoch, accuracy, dfa_accuracy, loss):
        with open(self._result_log, "a") as f:
            log = "\n epoch_num: %f, accuracy: %f, dfa_accuracy: %f, average_loss: %f \n" % (
                epoch, accuracy, dfa_accuracy, loss)
            f.write(log)

    def _calc_accuracy(self, source, ground_truth, prediction, weights):
        """
        Calculate the accuracy
        :param batch:
        :param predictions:
        :return: # correct
        """
        padded_seq = np.concatenate((prediction, np.array([0] * (len(ground_truth) - len(prediction)))), axis=0)
        diff = np.sum(np.abs((padded_seq - ground_truth) * weights))

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
        dfa_result = False
        try:
            gold = _replace(self._vocab_manager.decode(ground_truth))
            pred = _replace(self._vocab_manager.decode(padded_seq))
            out = subprocess.check_output(['java', '-jar', 'regex_dfa_equals.jar', gold, pred])
            dfa_result = '\n1' in out.decode()
        except Exception as e:
            print(e)
            dfa_result = False
        self._log_test(source, ground_truth, padded_seq, diff == 0, dfa_result)
        return diff == 0, dfa_result

    def test(self):
        # TODO Batch test
        """
        Test Model
        :return:
        """

        def _test(bs=1):
            sample = self._test_data_iterator.get_batch(1)
            last_predictions, _predictions, logprobs, mask, decoder_states, feed_dict = self._test_model.predict(sample.encoder_seq, sample.reverse_encoder_seq)
            last_predictions, _predictions, logprobs, mask, decoder_states = self._session.run(
                (last_predictions, _predictions, logprobs, mask, decoder_states), feed_dict=feed_dict)

            if np.sum(mask) == 0:
                index = np.argmax(logprobs)
            else:
                index = np.argmin(logprobs * (-mask))
            _predictions = _predictions[index]
            ground_truth = np.array(sample.target_seq[0])
            weights = np.array(sample.weights[0])
            exact_match, dfa_equality = self._calc_accuracy(sample.encoder_seq[0], ground_truth, _predictions, weights)
            return 1, exact_match, dfa_equality

        total = 0
        exact_correct = 0
        dfa_corrent = 0
        for i in range(self._test_data_iterator.size):
            t, e, d = _test()
            total += t
            exact_correct += e
            dfa_corrent += d
        return exact_correct/total, dfa_corrent/total

    def train(self):
        """
        Train model
        :return:
        """
        losses = list()

        def evaluate(epoch_num):
            nonlocal losses
            loss = np.average(np.array(losses))
            self._result_log = os.path.join(self._result_log_base_path, "epoch_%d.txt" % epoch_num)
            accuracy, dfa_accuracy = self.test()
            print("epoch_num: %f, accuracy: %f, dfa_accuracy: %f,  average_loss: %f " % (
                epoch_num, accuracy, dfa_accuracy, loss))
            self._log_epoch(epoch_num, accuracy, dfa_accuracy, loss)
            losses = list()

        self._train_data_iterator.epoch_cb = evaluate

        while self._train_data_iterator.epoch < self._epoches:
            batch = self._train_data_iterator.get_batch(self._batch_size)
            prediction, loss, optimizer, feed = self._train_model.train(batch)
            prediction, loss, optimizer = self._session.run((prediction, loss, optimizer,),
                                                            feed_dict=feed)
            losses.append(loss)
            # print(prediction, loss, optimizer)

    def run(self):
        self.train()
        """
        samples = self._test_data_iterator.get_batch(self._batch_size)

        print(samples._print())

        predictions, loss, optimizer, feed_dict = self._train_model.train(samples)

        predictions, loss, optimizer = self._session.run((predictions, loss, optimizer), feed_dict=feed_dict)

        print(predictions)
        print("=========================")
        print(loss)
        """
        self._session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--conf", help="Configuration File")
    args = parser.parse_args()

    runtime = ModelRuntime(args.conf)
    runtime.init_session()
    runtime.run()
