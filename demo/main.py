# coding=utf8

"""
Natural Language to Regular Expression Demo

Example:
    NL: "lines with the string <M0> before string <M1> followed by string <M2>"
    Regex: "( <M0> ) . * ( <M1> . * <M2> . * ) . *"
"""

import copy
import re
import os
import json
import tensorflow as tf
from pprint import pprint
from nltk.tokenize import ToktokTokenizer
from baseline_with_case_model import Model as BaselineWithCaseModel
from data_provider import VocabManager
from data_provider import Batch as BaselineWithCaseBatch
from ng_beam_search_model import Model as BeamSearchModel
from beam_search_model_data_provider import VocabManager as BeamSearchModelVocabManager
from beam_search_model_data_provider import Batch as BeamSearchModelBatch


KEYWORD_PATTERN = re.compile('\"(.*?)\"')
tokenizer = ToktokTokenizer()

MAX_CASE_NUM = 4
MAX_SENTENCE_LENGTH = 30
MAX_REGEX_LENGTH = 40
MAX_CASE_LENGTH = 100


def read_configuration(path):
    with open(path, "r") as f:
        return json.load(f)


def replace_keyword(string):
    """
    Replace Keywords in Natural Language, (Wrapped by double quote)
    :param string:
    :return:
    """
    keywords = KEYWORD_PATTERN.findall(string)

    replaced_dict = dict()
    s = copy.deepcopy(string)
    for kw, ph in zip(keywords, ["<M0>", "<M1>", "<M2>", "<M3>"]):
        s = s.replace('"'+kw+'"', ph)
        replaced_dict.update({
            '"' + kw + '"': ph
        })
    return s, replaced_dict


def replace_keywords_in_case(case, keywords_dict):
    """
    Replace keyword in Case
    :param case:
    :return:
    """
    c = copy.deepcopy(case)
    for kw in keywords_dict.keys():
        _kw = kw.replace('"', "")
        c = c.replace(_kw, keywords_dict[kw])
    return c


def process_sentence(sentence_vocab_manager, sentence):
    words = sentence.strip().split()
    ids = list()
    for word in words:
        ids.append(sentence_vocab_manager.word2id(word))
    ids.append(VocabManager.EOS_TOKEN_ID)
    sequence_length = len(ids)
    temp_length = len(ids)
    while temp_length < MAX_SENTENCE_LENGTH:
        ids.append(VocabManager.PADDING_TOKEN_ID)
        temp_length += 1
    return ids, sequence_length


def process_case(case_vocab_manager, case):
    words = case.strip().split()
    ids = list()
    for word in words:
        ids.append(case_vocab_manager.word2id(word))
    ids.append(VocabManager.EOS_TOKEN_ID)
    sequence_length = len(ids)
    temp_length = len(ids)

    while temp_length < MAX_CASE_LENGTH:
        ids.append(VocabManager.PADDING_TOKEN_ID)
        temp_length += 1
    return ids, sequence_length


def build_baseline_with_case_batch(sentence_vocab_manager, case_vocab_manager, sentence, cases):
    sentence_samples = list()
    sentence_length = list()
    case_samples = list()
    case_length = list()
    regex_length = list()
    regex_samples = list()
    # Remove GO TOKEN ID
    regex_targets = list()

    regex_targets.append([VocabManager.PADDING_TOKEN_ID]*MAX_REGEX_LENGTH)
    for i in range(len(cases)):
        sentence_word_ids = process_sentence(sentence_vocab_manager, sentence)
        sentence_samples.append(sentence_word_ids[0])
        sentence_length.append(sentence_word_ids[1])
        case_word_ids = process_case(case_vocab_manager, cases[i])
        case_samples.append(case_word_ids[0])
        case_length.append(case_word_ids[1])
        regex_samples.append([VocabManager.PADDING_TOKEN_ID]*MAX_REGEX_LENGTH)
        regex_length.append(MAX_REGEX_LENGTH)

    return BaselineWithCaseBatch(
        sentences=sentence_samples,
        cases=case_samples,
        sentence_length=sentence_length,
        case_length=case_length,
        regexs=regex_samples,
        regex_length=regex_length,
        regex_targets=regex_targets
    )


def build_baseline_with_case_model(
        sentence_vocab_manager,
        case_vocab_manager,
        regex_vocab_manager,
        configuration
    ):
    test_model = BaselineWithCaseModel(
        sentence_vocab_manager,
        case_vocab_manager,
        regex_vocab_manager,
        configuration,
        is_test=True,
        pretrained_sentence_embedding=None,
        pretrained_case_embedding=None
    )
    return test_model


def test_with_case(regex, cases):
    for c in cases:
        if not re.search(regex, c):
            return False
    return True


def build_beam_search_model_batch(vocab_manager, sentence):
    s = list()
    for word in tokenizer.tokenize(sentence):
        s.append(vocab_manager.encoder_word2id(word)["id"])
    s = s[::-1]
    s.append(BeamSearchModelVocabManager.EOS_TOKEN_ID)
    temp_len = len(s)
    while temp_len < MAX_SENTENCE_LENGTH:
        s.insert(0, BeamSearchModelVocabManager.PADDING_TOKEN_ID)
        temp_len += 1

    weight = [0.0] * MAX_REGEX_LENGTH
    regex = [BeamSearchModelVocabManager.PADDING_TOKEN_ID] * MAX_REGEX_LENGTH

    return BeamSearchModelBatch(
        source=[s],
        weight=[weight],
        target=[regex]
    )


def recover_regex(regex, keyword_dict):
    _ = copy.deepcopy(regex)
    for kw, ph in keyword_dict.items():
        _ = _.replace(ph, kw.replace('"', ""))
    _ = _.replace("<VOW>", 'AEIOUaeiou')
    _ = _.replace("<NUM>", '0-9')
    _ = _.replace("<LET>", 'A-Za-z')
    _ = _.replace("<CAP>", 'A-Z')
    _ = _.replace("<LOW>", 'a-z')
    return _


def main():

    sentence_vocab_manager = VocabManager(
        "vocab\\sentence_vocab.json"
    )
    case_vocab_manager = VocabManager(
        "vocab\\case_vocab.json"
    )
    regex_vocab_manager = VocabManager(
        "vocab\\regex_vocab.json"
    )

    beam_search_vocab_manager = BeamSearchModelVocabManager(
        "vocab\\beam_search_vocab.json"
    )

    baseline_with_case_graph = tf.Graph()
    baseline_with_case_session = tf.Session(graph=baseline_with_case_graph)
    baseline_with_case_configuration = read_configuration("baseline_with_case_checkpoint\\config.json")

    with baseline_with_case_session.graph.as_default():
        with tf.variable_scope("seq2seq") as scope:
            baseline_with_case_model = build_baseline_with_case_model(
                sentence_vocab_manager,
                case_vocab_manager,
                regex_vocab_manager,
                baseline_with_case_configuration
            )
            scope.reuse_variables()
            baseline_with_case_configuration["case_num"] = 2
            baseline_with_2_case_model = build_baseline_with_case_model(
                sentence_vocab_manager,
                case_vocab_manager,
                regex_vocab_manager,
                baseline_with_case_configuration
            )
            baseline_with_case_configuration["case_num"] = 3
            baseline_with_3_case_model = build_baseline_with_case_model(
                sentence_vocab_manager,
                case_vocab_manager,
                regex_vocab_manager,
                baseline_with_case_configuration
            )
            baseline_with_case_configuration["case_num"] = 4
            baseline_with_4_case_model = build_baseline_with_case_model(
                sentence_vocab_manager,
                case_vocab_manager,
                regex_vocab_manager,
                baseline_with_case_configuration
            )
            checkpoint_saver = tf.train.Saver(name="baseline_with_case_saver")
            checkpoint_saver.restore(baseline_with_case_session, "baseline_with_case_checkpoint\\tf_best_checkpoint")

    beam_search_graph = tf.Graph()
    beam_search_model_session = tf.Session(graph=beam_search_graph)
    beam_search_model_configuration = read_configuration("ng_beam_search_checkpoint\\config.json")

    with beam_search_model_session.graph.as_default():
        with tf.variable_scope("seq2seq") as scope:
            beam_search_model = BeamSearchModel(
                beam_search_vocab_manager,
                beam_search_model_configuration,
                is_test=True,
            )
            beam_search_model_checkpoint_saver = tf.train.Saver(name="beam_search_saver")
            beam_search_model_checkpoint_saver.restore(beam_search_model_session, "ng_beam_search_checkpoint\\tf_best_checkpoint")

    os.system("cls")
    print("Welcome to Natural Language to Regex System = =")
    while True:
        nl = input("Desired String pattern: ")
        # print("Original NL: %s" % nl)
        replaced_nl, replaced_keyword_dict = replace_keyword(nl)
        # print("Preprocessed NL: %s" % replaced_nl)

        is_case = input("Any positive cases y/n ?")

        if is_case.strip() == "y":
            cases = list()
            preprocessed_cases = list()
            # Use Baseline with case checkpoint
            for i in range(MAX_CASE_NUM):
                c = input("Cases %d: " % i)
                c = c.strip()
                if not c or len(c) == 0:
                    break
                cases.append(c)
                preprocessed_cases.append(replace_keywords_in_case(c, replaced_keyword_dict))
            # print(preprocessed_cases)

            batch = build_baseline_with_case_batch(
                sentence_vocab_manager=sentence_vocab_manager,
                case_vocab_manager=case_vocab_manager,
                sentence=replaced_nl,
                cases=preprocessed_cases
            )

            if len(preprocessed_cases) == 1:
                model = baseline_with_case_model
            elif len(preprocessed_cases) == 2:
                model = baseline_with_2_case_model
            elif len(preprocessed_cases) == 3:
                model = baseline_with_3_case_model
            else:
                model = baseline_with_4_case_model

            predictions, predictions_logprobs, feed_dict = model.predict(batch)
            predictions, logprobs = baseline_with_case_session.run((predictions, predictions_logprobs,), feed_dict)

            predicted_regexes = list()
            for p in predictions[0]:
                predicted_regexes.append(regex_vocab_manager.decode(p, delimiter=""))
            # pprint(predicted_regexes)

            recovered_truth_regex = list()
            for r in predicted_regexes:
                _ = recover_regex(r, replaced_keyword_dict)
                recovered_truth_regex.append(_)

            print("Passed Case Regex: ")
            passed_regex = list()
            for r in recovered_truth_regex:
                if test_with_case(r, cases):
                    passed_regex.append(r)
            pprint(passed_regex)
            print("=======================================")

        else:
            # Use NG beam search checkpoint
            batch = build_beam_search_model_batch(
                beam_search_vocab_manager,
                replaced_nl
            )

            last_predictions, _predictions, logprobs, mask, decoder_states, feed_dict = beam_search_model.predict(
                batch.encoder_seq)
            last_predictions, predictions, logprobs, mask, decoder_states = beam_search_model_session.run(
                (last_predictions, _predictions, logprobs, mask, decoder_states), feed_dict=feed_dict)

            predicted_regexes = list()
            for p in predictions:
                predicted_regexes.append(beam_search_vocab_manager.decode(p, delimiter=""))
            # pprint(predicted_regexes)

            recovered_truth_regex = list()
            for r in predicted_regexes:
                _ = recover_regex(r, replaced_keyword_dict)
                recovered_truth_regex.append(_)

            pprint(recovered_truth_regex)

        print("\n\n\n")

if __name__ == "__main__":
    main()
