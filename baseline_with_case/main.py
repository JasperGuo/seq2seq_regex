# coding=utf8

import argparse
import os
import numpy as np
import json
import time

import subprocess
from models import Model
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
            os.path.abspath(self._conf["sentence_vocab_files"])
        )
        self._case_vocab_manager = VocabManager(
            os.path.abspath(self._conf["case_vocab_files"])
        )
        self._regex_vocab_manager = VocabManager(
            os.path.abspath(self._conf["regex_vocab_files"])
        )

        self._epoches = self._conf["epoches"]
        self._batch_size = self._conf["batch_size"]
        self._max_sentence_length = self._conf["max_sentence_length"]
        self._max_case_length = self._conf["max_case_length"]
        self._max_regex_length = self._conf["max_regex_length"]
        self._case_num = self._conf["case_num"]
        self._is_pretrained_embedding_used = self._conf["is_pretrained_embedding_used"]
        self._sentence_word_pretrained_embedding_path = self._conf["sentence_word_pretrained_embedding_path"]
        self._case_word_pretrained_embedding_path = self._conf["case_word_pretrained_embedding_path"]
        self._is_beam_search = self._conf["is_beam_search"]
        self._beam_size = self._conf["beam_size"]

        if self._is_pretrained_embedding_used:
            self._sentence_embedding = self._load_pretrain_embedding(self._sentence_word_pretrained_embedding_path)
            self._case_embedding = self._load_pretrain_embedding(self._case_word_pretrained_embedding_path)
        else:
            self._sentence_embedding = None
            self._case_embedding = None

        self._curr_time = str(int(time.time()))
        self._log_dir = os.path.abspath(self._conf["log_dir"])

        self._result_log_base_path = os.path.abspath(os.path.join(self._conf["result_log"], self._curr_time))

        self._checkpoint_path = os.path.abspath(os.path.join(self._conf["checkpoint_path"], self._curr_time))
        self._checkpoint_file = os.path.join(self._checkpoint_path, "tf_checkpoint")
        self._best_checkpoint_file = os.path.join(os.path.curdir, self._checkpoint_path, "tf_best_checkpoint")

        os.mkdir(self._checkpoint_path)
        self._is_test_capability = self._conf["is_test_capability"]

        # Learning Rate Strategy
        self._default_learning_rate = self._conf["default_learning_rate"]
        self._learning_rate_decay_interval = self._conf["learning_rate_decay_interval"]
        self._learning_rate_decay_factor = self._conf["learning_rate_decay_factor"]

        self._train_data_iterator = DataIterator(
            os.path.abspath(self._conf["train_file"]),
            self._sentence_vocab_manager,
            self._case_vocab_manager,
            self._regex_vocab_manager,
            self._max_sentence_length,
            self._max_case_length,
            self._max_regex_length,
            self._batch_size,
            self._case_num
        )

        self._test_data_iterator = DataIterator(
            os.path.abspath(self._conf["test_file"]),
            self._sentence_vocab_manager,
            self._case_vocab_manager,
            self._regex_vocab_manager,
            self._max_sentence_length,
            self._max_case_length,
            self._max_regex_length,
            self._batch_size,
            self._case_num
        )

        self._development_data_iterator = DataIterator(
            os.path.abspath(self._conf["development_file"]),
            self._sentence_vocab_manager,
            self._case_vocab_manager,
            self._regex_vocab_manager,
            self._max_sentence_length,
            self._max_case_length,
            self._max_regex_length,
            self._batch_size,
            self._case_num
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
            f.write(json.dumps(self._conf, indent=4, sort_keys=True))

    def _load_pretrain_embedding(self, path):
        return np.load(path)

    def init_session(self, checkpoint=None):
        self._session = tf.Session()

        with tf.variable_scope("seq2seq") as scope:
            self._train_model = Model(
                self._sentence_vocab_manager,
                self._case_vocab_manager,
                self._regex_vocab_manager,
                self._conf,
                is_test=False,
                pretrained_sentence_embedding=self._sentence_embedding,
                pretrained_case_embedding=self._case_embedding
            )
            scope.reuse_variables()
            self._test_model = Model(
                self._sentence_vocab_manager,
                self._case_vocab_manager,
                self._regex_vocab_manager,
                self._conf,
                is_test=True,
                pretrained_sentence_embedding=self._sentence_embedding,
                pretrained_case_embedding=self._case_embedding
            )
            self._saver = tf.train.Saver()
            if not checkpoint:
                init = tf.global_variables_initializer()
                self._session.run(init)
            else:
                self._saver.restore(self._session, checkpoint)
            self._file_writer = tf.summary.FileWriter(self._log_dir, self._session.graph)

    def log(self, file, source, cases, ground_truth, prediction, diff, dfa_diff, score=0.0):
        source = 'S: ' + self._sentence_vocab_manager.decode(source)

        case_str = list()
        for case in cases:
            case_str.append('C: ' + self._case_vocab_manager.decode(case))
        case_str = '\n'.join(case_str)

        padded_seq_str = 'p: ' + self._regex_vocab_manager.decode(prediction)
        ground_truth_str = 'T: ' + self._regex_vocab_manager.decode(ground_truth)

        segmentation = '============================================='
        summary = 'exact_match_diff: %d' % diff
        dna_equality = 'dfa_equality: %d' % dfa_diff
        score_str = 'score: %f' % score

        string = '\n'.join(['\n', summary, dna_equality, score_str, source, case_str, padded_seq_str, ground_truth_str, segmentation])

        with open(file, "a") as f:
            f.write(string)

    def epoch_log(self, file, num_epoch, train_accuracy, dev_accuracy, test_accuracy, average_loss):
        with open(file, "a") as f:
            f.write("epoch: %d, train_accuracy: %f, dev_accuracy: %f, test_accuracy: %f, average_loss: %f\n" % (num_epoch, train_accuracy, dev_accuracy, test_accuracy, average_loss))

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

        gold = _replace(self._regex_vocab_manager.decode(ground_truth))
        pred = _replace(self._regex_vocab_manager.decode(prediction))
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

    def _calc_batch_accuracy(self, predictions, ground_truths, is_dfa_test=False):
        """
        Check accuracy in batch
        :return:
            list, list
        """
        exact_correct = list()
        dfa_correct = list()
        for (prediction, ground_truth) in zip(predictions, ground_truths):
            ground_truth = np.array(ground_truth)
            padded_seq = np.concatenate((prediction, np.array([0] * (len(ground_truth) - len(prediction)))), axis=0)
            e, d = self._calc_accuracy(ground_truth, padded_seq, is_dfa_test=is_dfa_test)
            exact_correct.append(int(e))
            dfa_correct.append(int(d))
        return exact_correct, dfa_correct

    def test(self, data_iterator, description, is_log=False):
        set_exact_match = 0
        set_dfa_match = 0
        total = 0
        file = os.path.join(self._result_log_base_path, "test_" + self._curr_time + ".log")
        tqdm.write("Testing...")
        for i in tqdm(range(data_iterator.batch_per_epoch)):
            batch = data_iterator.get_batch()

            if not self._is_beam_search:

                predictions, feed_dict = self._test_model.predict(batch)
                predictions = self._session.run(predictions, feed_dict)

                _predictions = predictions.reshape(
                    [batch.batch_size, self._max_regex_length]
                )

                exact_match, dfa_correct = self._calc_batch_accuracy(_predictions, batch.regex_targets)
                set_exact_match += sum(exact_match)
                set_dfa_match += sum(dfa_correct)
                total += batch.batch_size

                if is_log:
                    cases_idx = 0
                    for p, t, em, dc in zip(_predictions, batch.regex_targets, exact_match, dfa_correct):
                        s = batch.sentences[cases_idx:cases_idx+self._case_num][0]
                        cases = batch.cases[cases_idx:cases_idx + self._case_num]
                        self.log(file, s, cases, t, p, em, dc)
                        cases_idx += self._case_num

            else:

                predictions, predictions_logprobs, feed_dict = self._test_model.predict(batch)
                predictions, logprobs = self._session.run((predictions, predictions_logprobs,), feed_dict)

                _predictions = predictions.reshape(
                    [batch.batch_size, self._beam_size, self._max_regex_length]
                )

                _logprobs = np.reshape(
                    logprobs,
                    [self._batch_size, self._beam_size]
                )

                cases_idx = 0
                for p, probs, t in zip(_predictions, _logprobs, batch.regex_targets):
                    tiled_t = np.reshape(
                        np.tile(np.array(t), [self._beam_size]),
                        [self._beam_size, self._max_regex_length]
                    )
                    exact_match, dfa_correct = self._calc_batch_accuracy(
                        p, tiled_t
                    )
                    cases = batch.cases[cases_idx:cases_idx+self._case_num]
                    sentence = batch.sentences[cases_idx:cases_idx+self._case_num][0]
                    if sum(exact_match) > 0:
                        set_exact_match += 1
                    if sum(dfa_correct) > 0:
                        set_dfa_match += 1
                    total += 1

                    cases_idx += self._case_num

                    for _p, prob, em, dc in zip(p, probs, exact_match, dfa_correct):
                        self.log(file, sentence, cases, t, _p, em, dc, score=prob)

        accuracy = set_exact_match/total
        dfa_accuracy = set_dfa_match/total
        tqdm.write(', '.join([description, "exact_match_accuracy: %f, dfa_match_accuracy: %f" % (accuracy, dfa_accuracy)]))
        return accuracy, dfa_accuracy

    def train(self):
        best_accuracy = 0.
        epoch_log_file = os.path.join(self._result_log_base_path, "epoch_result.log")
        curr_learning_rate = self._default_learning_rate
        last_updated_epoch = 0
        for epoch in tqdm(range(self._epoches)):
            self._train_data_iterator.shuffle()
            losses = list()
            total = 0
            train_exact_match = 0
            for i in tqdm(range(self._train_data_iterator.batch_per_epoch)):
                batch = self._train_data_iterator.get_batch()
                batch.learning_rate = curr_learning_rate
                predictions, loss, optimizer, feed_dict = self._train_model.train(batch)
                predictions, loss, optimizer = self._session.run((
                    predictions, loss, optimizer,
                ), feed_dict=feed_dict)
                losses.append(loss)

                exact_match, dfa_correct = self._calc_batch_accuracy(predictions, batch.regex_targets)
                train_exact_match += sum(exact_match)
                total += batch.batch_size

            tqdm.write(np.array_str(np.array(losses)))
            average_loss = np.average(np.array(losses))

            tqdm.write("epoch: %d, loss: %f" % (epoch, average_loss))
            train_accuracy = train_exact_match/total
            tqdm.write(', '.join(["Train", "accuracy: %f" % train_accuracy]))

            self._test_data_iterator.shuffle()
            self._development_data_iterator.shuffle()

            development_exact_accuracy, development_dfa_accuracy = self.test(self._development_data_iterator, "Development", is_log=False)
            # test_exact_accuracy, test_dfa_accuracy = self.test(self._test_data_iterator, "Test")

            self.epoch_log(epoch_log_file, epoch, train_accuracy, development_exact_accuracy, 0.00, average_loss)

            # Decay Learning rate
            if epoch - last_updated_epoch >= self._learning_rate_decay_interval:
                curr_learning_rate = curr_learning_rate * self._learning_rate_decay_factor

            if development_exact_accuracy > best_accuracy:
                self._saver.save(self._session, self._best_checkpoint_file)
                best_accuracy = development_exact_accuracy
            tqdm.write("=================================================================")
        else:
            self._saver.save(self._session, self._checkpoint_file)

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

