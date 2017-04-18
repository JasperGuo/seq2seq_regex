# coding=utf8


import argparse
import os
import numpy as np
import json
import time
from model import Model
import tensorflow as tf
from data_provider import VocabManager, DataIterator
from tqdm import tqdm


def read_configuration(path):
    with open(path, "r") as f:
        return json.load(f)


class ModelRuntime:
    def __init__(self, configuration):
        self._conf = read_configuration(configuration)

        self._sentence_vocab_manager = VocabManager(self._conf["sentence_vocab_files"])
        self._case_vocab_manager = VocabManager(self._conf["case_vocab_files"])
        self._epoches = self._conf["epoches"]
        self._batch_size = self._conf["batch_size"]
        self._max_sentence_length = self._conf["max_sentence_length"]
        self._max_case_length = self._conf["max_case_length"]
        self._log_dir = self._conf["log_dir"]
        self._result_log_base_path = os.path.abspath(self._conf["result_log"] + str(int(time.time())))
        self._is_test_capability = self._conf["is_test_capability"]

        self._train_data_iterator = DataIterator(
            self._conf["train_file"],
            self._sentence_vocab_manager,
            self._case_vocab_manager,
            self._max_sentence_length,
            self._max_case_length,
            self._batch_size
        )

        self._test_data_iterator = DataIterator(
            self._conf["test_file"],
            self._sentence_vocab_manager,
            self._case_vocab_manager,
            self._max_sentence_length,
            self._max_case_length,
            self._batch_size
        )

        self._development_data_iterator = DataIterator(
            self._conf["development_file"],
            self._sentence_vocab_manager,
            self._case_vocab_manager,
            self._max_sentence_length,
            self._max_case_length,
            self._batch_size
        )

        os.mkdir(self._result_log_base_path)
        self._save_conf_file()

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
            self._train_model = Model(
                self._sentence_vocab_manager,
                self._case_vocab_manager,
                self._conf,
                is_test=False
            )
            scope.reuse_variables()
            self._test_model = Model(
                self._sentence_vocab_manager,
                self._case_vocab_manager,
                self._conf,
                is_test=True
            )
            init = tf.global_variables_initializer()
            self._session.run(init)
            self._file_writer = tf.summary.FileWriter(self._log_dir, self._session.graph)

    def test(self, data_iterator, description):
        total_error = 0
        total = 0
        for i in range(data_iterator.batch_per_epoch):
            batch = data_iterator.get_batch()
            predictions, feed_dict = self._test_model.predict(batch)
            predictions = self._session.run(predictions, feed_dict)
            ground_truth_labels = batch.labels
            total_error += np.sum(np.abs(predictions - ground_truth_labels))
            total += batch.batch_size
        tqdm.write(', '.join([description, "accuracy: %f" % (1-(total_error/total))]))

    def train(self):
        for epoch in tqdm(range(self._epoches)):
            self._train_data_iterator.shuffle()
            losses = list()
            total_errors = 0
            total = 0
            for i in tqdm(range(self._train_data_iterator.batch_per_epoch)):
                batch = self._train_data_iterator.get_batch()
                predictions, loss, optimizer, feed_dict = self._train_model.train(batch)
                predictions, loss, optimizer = self._session.run((
                    predictions, loss, optimizer
                ), feed_dict)
                losses.append(loss)
                ground_truth_labels = batch.labels

                total_errors += np.sum(np.abs(predictions - ground_truth_labels))
                total += batch.batch_size
            average_loss = np.average(np.array(losses))

            tqdm.write("epoch: %d, loss: %f" % (epoch, average_loss))
            tqdm.write(', '.join(["Train", "accuracy: %f" % (1 - (total_errors / total))]))
            self._test_data_iterator.shuffle()
            self._development_data_iterator.shuffle()
            self.test(self._development_data_iterator, "Development")
            self.test(self._test_data_iterator, "Test")
            tqdm.write("=================================================================")

    def run(self):
        self.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--conf", help="Configuration File")
    args = parser.parse_args()

    runtime = ModelRuntime(args.conf)
    runtime.init_session()
    runtime.run()
