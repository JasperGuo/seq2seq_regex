# coding=utf8

import os
import json
import numpy as np
from pprint import pprint
from data_iterator import VocabManager

TRAIN = os.path.join(os.pardir, "data", "train.json")
DEVELOPMENT = os.path.join(os.pardir, "data", "development.json")
TEST = os.path.join(os.pardir, "data", "test.json")
VOCAB = os.path.join(os.pardir, "data", "vocab.json")

FEED_TRAIN = os.path.join(os.pardir, "feed_tf", "train.json")
FEED_DEVELOPMENT = os.path.join(os.pardir, "feed_tf", "development.json")
FEED_TEST = os.path.join(os.pardir, "feed_tf", "test.json")
FEED_VOCAB = os.path.join(os.pardir, "feed_tf", "vocab.json")


def generate_feed_vocab(min_freq=2):
    """
    Merge vocab, filter those less than 1
    :param min_freq:
    :return:
    """
    with open(VOCAB, "r") as f:
        vocabs = json.load(f)

    results = dict()

    def _generate(vocab):
        result = dict()
        length = VocabManager.USEFUL_VOCAB_INIT_ID
        for word, frequency in vocab.items():
            if frequency < min_freq:
                continue
            result[word] = {
                "frequency": frequency,
                "id": length
            }
            length += 1
        result[VocabManager.UNKNOWN_TOKEN] = {
            "frequency": 0,
            "id": VocabManager.UNKNOWN_TOKEN_ID
        }
        result[VocabManager.PADDING_TOKEN] = {
            "frequency": 0,
            "id": VocabManager.PADDING_TOKEN_ID,
        }
        result[VocabManager.GO_TOKEN] = {
            "frequency": 0,
            "id": VocabManager.GO_TOKEN_ID
        }
        result[VocabManager.EOS_TOKEN] = {
            "frequency": 0,
            "id": VocabManager.EOS_TOKEN_ID
        }
        return result

    iteration = ["source", "target"]

    for v in iteration:
        results[v] = _generate(vocabs[v])

    with open(FEED_VOCAB, "w") as f:
        f.write(json.dumps(results, indent=4))

    return results


def replace_word_with_id(file, targ_file, vocab):
    """
    Replace_word_with_id
    :param file:
    :param vocab:
    :return:
    """

    with open(file, "r") as f:
        data = json.load(f)

    def _replace(words, v):
        r = list()
        words = words.strip().split()
        for word in words:
            if word not in v:
                r.append(VocabManager.UNKNOWN_TOKEN_ID)
            else:
                r.append(v[word]["id"])
        return r

    for pair in data:
        sentence = pair["sentence"]
        regex = pair["regex"]

        pair["source"] = _replace(sentence, vocab["source"])
        pair["target"] = _replace(regex, vocab["target"])

    with open(targ_file, "w") as f:
        f.write(json.dumps(data, indent=4))


if __name__ == "__main__":
    vocab = generate_feed_vocab()

    replace_word_with_id(TRAIN, FEED_TRAIN, vocab)
    replace_word_with_id(DEVELOPMENT, FEED_DEVELOPMENT, vocab)
    replace_word_with_id(TEST, FEED_TEST, vocab)
