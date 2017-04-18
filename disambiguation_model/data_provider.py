# coding=utf8

import json
import util


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
        return util.get_value(self._vocab, word, self.UNKNOWN_TOKEN)

    def id2word(self, wid):
        return util.get_value(self._vocab_id2word, wid)

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
