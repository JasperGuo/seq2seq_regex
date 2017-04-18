# coding=utf8
import argparse

from util import read_data, save_data


def collect_vocab(file_path, ts_path, tc_path, vocab_path):
    data = read_data(file_path)
    vocab = dict()
    for sample in data:
        case = sample["case"]
        words = case.split()
        for word in words:
            if word not in vocab:
                vocab[word] = 0
            vocab[word] += 1
    original_vocab = read_data(vocab_path)

    new_vocab = original_vocab["source"]
    save_data(ts_path, new_vocab)

    new_vocab = vocab
    save_data(tc_path, new_vocab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", help="source file")
    parser.add_argument("--ts", help="target sentence vocab file")
    parser.add_argument("--tc", help="target case vocab file")
    parser.add_argument("--vocab", help="vocab file")

    args = parser.parse_args()

    collect_vocab(args.source, args.ts, args.tc, args.vocab)