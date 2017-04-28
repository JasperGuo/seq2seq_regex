# coding=utf8

import hypertools as hyp
import argparse
import numpy as np
import json
import os


def read_vocab(vocab_path):
    with open(vocab_path, "r") as f:    
        vocab = json.load(f)
    return vocab


def visualize(vocab_path, embedding_path):
    """
    Visualize embedding, label datapoint from vocab
    :param vocab_path:
    :param embedding_path:
    :return:
    """

    embedding = np.load(embedding_path)
    print(embedding.shape)

    vocab = read_vocab(vocab_path)

    word_labels = list()
    for word, value in vocab.items():
        word_labels.append((word, value["id"]))
    word_labels.sort(key=lambda x: x[1])

    # Remove pad
    word_labels = word_labels[1:]

    datapoint_labels = [word[0] for word in word_labels]

    hyp.plot(embedding, 'o', labels=datapoint_labels, ndims=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab", help="Vocab file path", required=True)
    parser.add_argument("--embedding", help="Embedding file path", required=True)
    args = parser.parse_args()

    visualize(args.vocab, args.embedding)
