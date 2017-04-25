# coding=utf8

import sys
sys.path += [".."]
import argparse
import os
import numpy as np
import json
import time
from models.model import Model
import tensorflow as tf
from data_provider import VocabManager, DataIterator
from tqdm import tqdm


def read_configuration(path):
    with open(path, "r") as f:
        return json.load(f)


class ModelRuntime:

    def __init__(self, configuration):
        self._conf = read_configuration(configuration)

        self._sentence_vocab_manager = VocabManager(
            os.path.abspath(os.path.join(
                os.path.pardir,
                self._conf["sentence_vocab_files"]
            ))
        )
        self._case_vocab_manager = VocabManager(
            os.path.abspath(os.path.join(
                os.path.pardir,
                self._conf["case_vocab_files"]
            ))
        )
        self._epoches = self._conf["epoches"]
        self._batch_size = self._conf["batch_size"]
        self._max_sentence_length = self._conf["max_sentence_length"]
        self._max_case_length = self._conf["max_case_length"]
        self._curr_time = str(int(time.time()))
        self._log_dir = os.path.abspath(os.path.join(os.path.pardir, self._conf["log_dir"]))
        self._result_log_base_path = os.path.abspath(os.path.join(os.path.pardir, self._conf["result_log"] + self._curr_time))
        self._checkpoint_path = os.path.abspath(os.path.join(os.path.pardir, self._conf["checkpoint_path"] + self._curr_time))
        self._checkpoint_file = os.path.join(self._checkpoint_path, "tf_checkpoint")
        os.mkdir(self._checkpoint_path)
        self._is_test_capability = self._conf["is_test_capability"]

        self._train_data_iterator = DataIterator(
            os.path.abspath(os.path.join(os.path.pardir, self._conf["train_file"])),
            self._sentence_vocab_manager,
            self._case_vocab_manager,
            self._max_sentence_length,
            self._max_case_length,
            self._batch_size
        )

        self._test_data_iterator = DataIterator(
            os.path.abspath(os.path.join(os.path.pardir, self._conf["test_file"])),
            self._sentence_vocab_manager,
            self._case_vocab_manager,
            self._max_sentence_length,
            self._max_case_length,
            self._batch_size
        )

        self._development_data_iterator = DataIterator(
            os.path.abspath(os.path.join(os.path.pardir, self._conf["development_file"])),
            self._sentence_vocab_manager,
            self._case_vocab_manager,
            self._max_sentence_length,
            self._max_case_length,
            self._batch_size
        )

        os.mkdir(self._result_log_base_path)
        self._save_conf_file(self._result_log_base_path)
        self._save_conf_file(self._checkpoint_path)

    def _save_conf_file(self, base_path):
        """
        Save Configuration to result log directory
        :return:
        """
        path = os.path.join(base_path, "config.json")
        with open(path, "w") as f:
            f.write(json.dumps(self._conf, indent=4))

    def init_session(self, checkpoint=None):
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
            self._saver = tf.train.Saver()
            if not checkpoint:
                init = tf.global_variables_initializer()
                self._session.run(init)
            else:
                self._saver.restore(self._session, checkpoint)
            self._file_writer = tf.summary.FileWriter(self._log_dir, self._session.graph)

    def test(self, data_iterator, description, is_log=False):
        total_error = 0
        total = 0
        file = os.path.join(self._result_log_base_path, "test_" + self._curr_time + ".log")
        positive_vector_representation_file = os.path.join(self._result_log_base_path,
                                                           "positive_vector_" + self._curr_time + ".npy")
        negative_vector_representation_file = os.path.join(self._result_log_base_path,
                                                           "negative_vector_" + self._curr_time + ".npy")
        positive_vector_representation = None
        negative_vector_representation = None
        for i in range(data_iterator.batch_per_epoch):
            batch = data_iterator.get_batch()
            predictions, outputs, feed_dict = self._test_model.predict(batch)
            predictions, outputs = self._session.run((predictions, outputs), feed_dict)
            ground_truth_labels = batch.labels
            total_error += np.sum(np.abs(predictions - ground_truth_labels))
            total += batch.batch_size

            if is_log:

                temp_positive = list()
                temp_negative = list()
                for idx, label in enumerate(ground_truth_labels):
                    if label == 1:
                        temp_positive.append(outputs[idx])
                    else:
                        temp_negative.append(outputs[idx])
                if len(temp_positive) > 0:
                    temp_positive_array = np.array(temp_positive)
                    if not isinstance(positive_vector_representation, np.ndarray):
                        positive_vector_representation = temp_positive_array
                    else:
                        positive_vector_representation = np.concatenate((positive_vector_representation, temp_positive_array), axis=0)

                if len(temp_negative) > 0:
                    temp_negative_array = np.array(temp_negative)
                    if not isinstance(negative_vector_representation, np.ndarray):
                        negative_vector_representation = temp_negative_array
                    else:
                        negative_vector_representation = np.concatenate(
                            (negative_vector_representation, temp_negative_array), axis=0)

                self.log(file, batch, predictions)

        if is_log:
            np.save(positive_vector_representation_file, positive_vector_representation)
            np.save(negative_vector_representation_file, negative_vector_representation)

        accuracy = (1-(total_error/total))
        tqdm.write(', '.join([description, "accuracy: %f" % accuracy]))
        return accuracy

    def log(self, file, batch, predictions):
        with open(file, "a") as f:

            string = list()

            for i in range(batch.batch_size):
                sentence = self._sentence_vocab_manager.decode(batch.sentences[i])
                case = self._case_vocab_manager.decode(batch.cases[i], delimiter="")
                label = str(batch.labels[i])
                prediction = str(predictions[i])
                string.append(
                    '\n'.join([
                        "sentence: " + sentence,
                        "case: " + case,
                        "ground_truth: " + label,
                        "prediction: " + prediction,
                        "\n",
                        "================================================="
                    ])
                )
            f.write('\n'.join(string))

    def epoch_log(self, file, num_epoch, train_accuracy, dev_accuracy, test_accuracy, average_loss):
        with open(file, "a") as f:
            f.write("epoch: %d, train_accuracy: %f, dev_accuracy: %f, test_accuracy: %f, average_loss: %f\n" % (num_epoch, train_accuracy, dev_accuracy, test_accuracy, average_loss))

    def train(self):
        best_accuracy = 0.
        epoch_log_file = os.path.join(self._result_log_base_path, "epoch_result.log")
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
            train_accuracy = 1 - (total_errors / total)
            tqdm.write(', '.join(["Train", "accuracy: %f" % train_accuracy]))
            self._test_data_iterator.shuffle()
            self._development_data_iterator.shuffle()
            development_accuracy = self.test(self._development_data_iterator, "Development")
            test_accuracy = self.test(self._test_data_iterator, "Test")

            self.epoch_log(epoch_log_file, epoch, train_accuracy, development_accuracy, test_accuracy, average_loss)

            if development_accuracy > best_accuracy:
                self._saver.save(self._session, self._checkpoint_file)
                best_accuracy = development_accuracy
            tqdm.write("=================================================================")

    def run(self, is_test=False, is_log=False):
        if not is_test:
            self.train()
        else:
            self.test(self._test_data_iterator, "Test", is_log)
        self._session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--conf", help="Configuration File")
    parser.add_argument("--checkpoint", help="Is Checkpoint ? Then checkpoint path ?", required=False)
    parser.add_argument("--test", help="Is test ?", dest="is_test", action="store_true")
    parser.add_argument("--no-test", help="Is test ?", dest="is_test", action="store_false")
    parser.set_defaults(is_test=False)
    parser.add_argument("--log", help="Is log ?", dest="is_log", action="store_true")
    parser.add_argument("--no-log", help="Is log ?", dest="is_log", action="store_false")
    parser.set_defaults(is_log=False)
    args = parser.parse_args()

    print(args.conf, args.checkpoint, args.is_test, args.is_log)

    runtime = ModelRuntime(args.conf)
    runtime.init_session(args.checkpoint)
    runtime.run(args.is_test, args.is_log)

