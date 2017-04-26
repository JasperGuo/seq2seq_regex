# coding=utf8

import json
import util
import math
import random


class VocabManager:
    """
    Vocabulary Manager
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
        return data

    def __init__(self, vocab_path):
        self._vocab = self._read(vocab_path)

        self._vocab_id2word = dict()
        for (vocab, value) in self._vocab.items():
            self._vocab_id2word[value["id"]] = vocab

    @property
    def vocab(self):
        return self._vocab

    @property
    def vocab_len(self):
        return len((self._vocab.keys()))

    def word2id(self, word):
        return util.get_value(self._vocab, word, {"id": self.UNKNOWN_TOKEN_ID})["id"]

    def id2word(self, wid):
        return util.get_value(self._vocab_id2word, wid, self.UNKNOWN_TOKEN)

    def decode(self, wids, delimiter=" "):
        words = list()
        for wid in wids:
            if wid == self.PADDING_TOKEN_ID or wid == self.UNKNOWN_TOKEN_ID or wid == self.EOS_TOKEN_ID:
                continue
            words.append(self.id2word(wid))
        return delimiter.join(words)

    @classmethod
    def build_vocab(cls, file, target_file, min_freq=1):
        source_vocab = cls._read(file)

        def _generate(vocab):
            result = dict()
            length = cls.USEFUL_VOCAB_INIT_ID
            for word, frequency in vocab.items():
                if frequency < min_freq:
                    continue
                result[word] = {
                    "frequency": frequency,
                    "id": length
                }
                length += 1
            result[cls.UNKNOWN_TOKEN] = {
                "frequency": 0,
                "id": cls.UNKNOWN_TOKEN_ID
            }
            result[cls.PADDING_TOKEN] = {
                "frequency": 0,
                "id": cls.PADDING_TOKEN_ID,
            }
            result[cls.GO_TOKEN] = {
                "frequency": 0,
                "id": cls.GO_TOKEN_ID
            }
            result[cls.EOS_TOKEN] = {
                "frequency": 0,
                "id": cls.EOS_TOKEN_ID
            }
            return result

        results = _generate(source_vocab)

        with open(target_file, "w") as f:
            f.write(json.dumps(results, indent=4))


class Batch:
    """
    Batch Data
    """
    def __init__(self, sentences, cases, sentence_length, case_length, labels):
        self.sentences = sentences
        self.cases = cases
        self.sentence_length = sentence_length
        self.case_length = case_length
        self.sentence_masks = list()
        sentence_max_length = len(sentences[0])
        for l in sentence_length:
            self.sentence_masks.append(
                [1]*l + [0]*(sentence_max_length-l)
            )
        self.case_masks = list()
        case_max_length = len(cases[0])
        for l in case_length:
            self.case_masks.append(
                [1] * l + [0] * (case_max_length - l)
            )

        self.labels = labels

    @property
    def batch_size(self):
        return len(self.sentence_length)

    def _print(self):
        print(self.sentences)
        print(self.sentence_length)
        print(self.cases)
        print(self.case_length)
        print(self.labels)


class DataIterator:

    def __init__(self, data_path, sentence_vocab, case_vocab, max_sentence_len, max_case_len, batch_size):
        self._cursor = 0
        self._max_sentence_len = max_sentence_len
        self._max_case_len = max_case_len
        self._sentence_vocab = sentence_vocab
        self._case_vocab = case_vocab
        self._data = self._read_data(data_path)

        # Remove the training examples that are too long
        rm_list = list()
        for value in self._data:
            sentence = value["sentence"]
            case = value["case"]
            if len(sentence) > self._max_sentence_len or len(case) > self._max_case_len:
                rm_list.append(value)
        for r in rm_list:
            self._data.remove(r)

        self._size = len(self._data)
        self._batch_size = batch_size
        self._batch_per_epoch = math.floor(self._size / self._batch_size)
        self.shuffle()

    @property
    def size(self):
        return self._size

    @property
    def batch_per_epoch(self):
        return self._batch_per_epoch

    def shuffle(self):
        random.shuffle(self._data)
        self._cursor = 0

    def _read_data(self, data_path):
        with open(data_path, "r") as f:
            data = json.load(f)
        new_data = list()
        for sample in data:
            sentence, sentence_length = self.process_sentence(sample["sentence"])
            case, case_length = self.process_case(sample["case"])
            new_data.append({
                "sentence": sentence,
                "label": sample["label"],
                "case": case,
                "sentence_length": sentence_length,
                "case_length": case_length
            })
        return new_data

    def process_sentence(self, sentence):

        words = sentence.strip().split()
        ids = list()
        for word in words:
            ids.append(self._sentence_vocab.word2id(word))
        # ids.append(VocabManager.EOS_TOKEN_ID)
        sequence_length = len(ids)
        temp_length = len(ids)
        while temp_length < self._max_sentence_len:
            ids.append(VocabManager.PADDING_TOKEN_ID)
            temp_length += 1
        return ids, sequence_length

    def process_case(self, case):
        words = case.strip().split()
        ids = list()
        for word in words:
            ids.append(self._case_vocab.word2id(word))
        # ids.append(VocabManager.EOS_TOKEN_ID)
        sequence_length = len(ids)
        temp_length = len(ids)

        while temp_length < self._max_case_len:
            ids.append(VocabManager.PADDING_TOKEN_ID)
            temp_length += 1
        return ids, sequence_length

    def get_batch(self):

        if self._cursor + self._batch_size > self._size:
            raise IndexError("Index Error")

        samples = self._data[self._cursor:self._cursor+self._batch_size]
        self._cursor += self._batch_size
        sentence_samples = [s["sentence"] for s in samples]
        case_samples = [s["case"] for s in samples]
        sentence_length = [s["sentence_length"] for s in samples]
        case_length = [s["case_length"] for s in samples]
        labels = [s["label"] for s in samples]
        return Batch(
            sentences=sentence_samples,
            cases=case_samples,
            sentence_length=sentence_length,
            case_length=case_length,
            labels=labels
        )