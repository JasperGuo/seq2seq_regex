# coding=utf8

"""
Data Iterator for batch data
"""

import sys

sys.path += [".."]

import util
import json
import math
import random


class VocabManager:
    """
    Vocabulary Manager for input and output tokens
    """
    PADDING_TOKEN_ID = 0
    UNKNOWN_TOKEN_ID = 1
    GO_TOKEN_ID = 2
    EOS_TOKEN_ID = 3

    USEFUL_VOCAB_INIT_ID = 4

    UNKNOWN_TOKEN = "<U>"
    PADDING_TOKEN = "<P>"
    GO_TOKEN = "<GO>"
    EOS_TOKEN = "<EOS>"

    @staticmethod
    def _read(vocab_path):
        with open(vocab_path, "r") as f:
            data = json.load(f)
        return data["source"], data["target"]

    def __init__(self, vocab_path):
        (self._source, self._target) = self._read(vocab_path)

        self._vocab_source_id2word = dict()
        for (vocab, value) in self._source.items():
            self._vocab_source_id2word[value["id"]] = vocab

        self._vocab_target_id2word = dict()
        for (vocab, value) in self._target.items():
            self._vocab_target_id2word[value["id"]] = vocab

    @property
    def source(self):
        return self.source

    @property
    def vocab_source_len(self):
        return len((self._source.keys()))

    @property
    def vocab_target(self):
        return self._target

    @property
    def vocab_target_len(self):
        return len((self._target.keys()))

    def encoder_word2id(self, word):
        return util.get_value(self._source, word, {"id": self.UNKNOWN_TOKEN_ID})

    def decoder_word2id(self, word):
        return util.get_value(self._target, word, {"id": self.UNKNOWN_TOKEN_ID})

    def encoder_id2word(self, wid):
        return util.get_value(self._vocab_source_id2word, wid)

    def decoder_id2word(self, wid):
        return util.get_value(self._vocab_target_id2word, wid)

    def decode(self, ids, delimiter=""):
        result = list()
        for wid in ids:
            wid = int(wid)
            if wid in [self.PADDING_TOKEN_ID, self.EOS_TOKEN_ID, self.GO_TOKEN_ID]:
                continue
            result.append(self.decoder_id2word(wid))
        return delimiter.join(result)

    def decode_source(self, ids, delimiter=" "):
        result = list()
        for wid in ids:
            wid = int(wid)
            if wid in [self.PADDING_TOKEN_ID, self.EOS_TOKEN_ID, self.GO_TOKEN_ID]:
                continue
            result.append(self.encoder_id2word(wid))
        return delimiter.join(reversed(result))


class Batch:
    """
    Batch Data
    """

    def __init__(self, source, target, weight):
        self._encoder_seqs = source
        self._decoder_seqs = target
        self._weights = weight

        self._target_seqs = list()

        for i in target:
            self._target_seqs.append(i[1:] + [VocabManager.PADDING_TOKEN_ID])

    @property
    def encoder_seq(self):
        return self._encoder_seqs

    @property
    def decoder_seq(self):
        return self._decoder_seqs

    @property
    def target_seq(self):
        return self._target_seqs

    @property
    def weights(self):
        return self._weights

    @property
    def batch_size(self):
        return 0 if not self._encoder_seqs else len(self._encoder_seqs)

    def _print(self):
        """
        test only
        :return:
        """
        print(self._encoder_seqs)
        print(self._decoder_seqs)
        print(self._target_seqs)
        print(self._weights)

        length = []
        for s in self._encoder_seqs:
            length.append(len(s))
        for y in self._decoder_seqs:
            length.append(len(y))
        print(length)


class DataIterator:

    def _read_data(self, data_path):
        with open(data_path, "r") as f:
            data = json.load(f)

        for pair in data:
            pair["source"] = self.process_x(pair["source"])
            pair["target"] = self.process_y(pair["target"])
            pair["weight"] = self.process_weight(pair["target"])

        return data

    def __init__(self, data, max_x_len, max_y_len, batch_size):
        """
        :param data: json file path
        :param max_x_len:
        :param max_y_len:
        :param epoch_cb: When accomplish an epoch, call the function with #epoch
        """
        self._cursor = 0
        self._epochs = 0
        self._max_x_len = max_x_len
        self._max_y_len = max_y_len
        self._data = self._read_data(data)

        # Remove the training examples that are too long
        rm_list = list()
        for value in self._data:
            source = value["source"]
            target = value["target"]
            if len(source) > self._max_x_len or len(target) > self._max_y_len:
                rm_list.append(value)
        for r in rm_list:
            self._data.remove(r)

        self._size = len(self._data)
        self._batch_size = batch_size
        self._batch_per_epoch = math.floor(self._size / self._batch_size)

    @property
    def batch_per_epoch(self):
        return self._batch_per_epoch

    @property
    def epoch(self):
        return self._epochs

    def shuffle(self):
        random.shuffle(self._data)
        self._cursor = 0

    @property
    def size(self):
        return self._size

    def print(self):
        """
        used only test
        :return:
        """
        print(self._data)

    def process_x(self, x):
        """
        1. Reverse the input sequence, append <EOS> at the end of the sequence
        2. Padding <PAD>
        :param X:
        :return:
        """
        temp = x[::-1]
        temp.append(VocabManager.EOS_TOKEN_ID)
        temp_len = len(temp)
        while temp_len < self._max_x_len:
            temp.insert(0, VocabManager.PADDING_TOKEN_ID)
            temp_len += 1
        return temp

    def process_y(self, y):
        """
        1. Insert <GO> at the beginning of the sequence
        2. Append <EOS> at the end of the sequence
        3. Pad <PAD> to max length
        :param Y:
        :return:
        """
        temp = y
        temp.insert(0, VocabManager.GO_TOKEN_ID)
        temp.append(VocabManager.EOS_TOKEN_ID)
        temp_len = len(temp)
        while temp_len < self._max_y_len:
            temp.append(VocabManager.PADDING_TOKEN_ID)
            temp_len += 1
        return temp

    def process_weight(self, y):
        """
        Calc weight for each token in y
        :param y:
        :return:
        """
        try:
            first_pad_idx = y.index(VocabManager.PADDING_TOKEN_ID)
        except ValueError:
            return [1.0] * len(y)
        else:
            return [1.0] * first_pad_idx + [0.0] * (self._max_y_len - first_pad_idx)

    def get_batch(self):
        """
        :param n:                  batch size
        :return: source samples, target samples
        """
        
        if self._cursor + self._batch_size > self._size:
            raise IndexError("Index Error")

        samples = self._data[self._cursor:self._cursor + self._batch_size]
        self._cursor += self._batch_size
        source_sample = [s["source"] for s in samples]
        target_sample = [s["target"] for s in samples]
        weights = [s["weight"] for s in samples]
        return Batch(source_sample, target_sample, weights)
