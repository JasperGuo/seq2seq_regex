# coding=utf8

# coding=utf8

"""
Conditional Classification model, encode the test case conditioned on the sentence
"""
import sys

sys.path += [".."]
import util
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell


def softmax_with_mask(tensor, mask):
    """
    Calculate Softmax with mask
    :param tensor: [shape1, shape2]
    :param mask:   [shape1, shape2]
    :return:
    """
    exp_tensor = tf.exp(tensor)
    masked_exp_tensor = tf.multiply(exp_tensor, mask)
    total = tf.reshape(
        tf.reduce_sum(masked_exp_tensor, axis=1),
        shape=[-1, 1]
    )
    return tf.div(masked_exp_tensor, total)


class Model:
    epsilon = 1e-5

    def __init__(self, sentence_vocab_manager, case_vocab_manager, regex_vocab_manager, opts,
                 is_test=False, pretrained_sentence_embedding=None, pretrained_case_embedding=None):
        self._sentence_vocab_manager = sentence_vocab_manager
        self._case_vocab_manager = case_vocab_manager
        self._regex_vocab_manager = regex_vocab_manager

        self._hidden_dim = util.get_value(opts, "hidden_dim", 150)
        self._layers = util.get_value(opts, "layer", 2)
        self._max_sentence_length = util.get_value(opts, "max_sentence_length", 30)
        self._max_case_length = util.get_value(opts, "max_case_length", 100)
        self._max_regex_length = util.get_value(opts, "max_regex_length", 40)
        self._embedding_dim = util.get_value(opts, "embedding_dim", 150)
        self._learning_rate = util.get_value(opts, "learning_rate", 0.01)
        self._gradient_clip = util.get_value(opts, "gradient_clip", 5)
        self._dropout = util.get_value(opts, "dropout", 0.25)

        self._batch_size = util.get_value(opts, "batch_size", 5)
        self._case_num = util.get_value(opts, "case_num", 5)

        self._is_embedding_fine_tune = util.get_value(opts, "is_embedding_fine_tuned", True)

        self._actual_batch_size = self._batch_size * self._case_num

        self._is_test = is_test

        self._pretrained_sentence_embedding = pretrained_sentence_embedding
        self._pretrained_case_embedding = pretrained_case_embedding

        if self._is_test:
            self._batch_size = 1
            self._actual_batch_size = self._case_num

        if not self._is_test:
            self._build_train_graph()
        else:
            self._build_test_graph()

    def _build_sentence_embedding_layer(self, pretrained_sentence_word_embedding=None):
        """
        Build sentence word embedding
        :param pretrained_sentence_word_embedding:
        :return:
        """
        with tf.variable_scope("sentence_embedding_layer"):
            pad_embeddings = tf.get_variable(
                initializer=tf.zeros([1, self._embedding_dim]),
                name="sentence_pad_embedding",
                trainable=False
            )
            if not isinstance(pretrained_sentence_word_embedding, np.ndarray):
                sentence_embedding = tf.get_variable(
                    initializer=tf.truncated_normal(
                        [self._sentence_vocab_manager.vocab_len - 1, self._embedding_dim],
                        stddev=0.5
                    ),
                    name='sentence_embedding'
                )

            else:
                # use pretrained sentence word embedding
                sentence_embedding = tf.get_variable(
                    name="sentence_embedding",
                    trainable=self._is_embedding_fine_tune,
                    shape=[self._sentence_vocab_manager.vocab_len, self._embedding_dim],
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(pretrained_sentence_word_embedding)
                )
            sentence_embedding = tf.concat(values=[pad_embeddings, sentence_embedding], axis=0)
            return sentence_embedding

    def _build_case_embedding_layer(self, pretrained_case_word_embedding=None):
        """
        Build test case embedding
        :param pretrained_case_word_embedding:
        :return:
        """
        with tf.variable_scope("case_embedding_layer"):
            pad_embeddings = tf.get_variable(
                initializer=tf.zeros([1, self._embedding_dim]),
                name="case_pad_embedding",
                trainable=False
            )
            if not isinstance(pretrained_case_word_embedding, np.ndarray):
                case_embedding = tf.get_variable(
                    initializer=tf.truncated_normal(
                        [self._case_vocab_manager.vocab_len - 1, self._embedding_dim],
                        stddev=0.5
                    ),
                    name='case_embedding'
                )
            else:
                # use pretrained case word embedding
                case_embedding = tf.get_variable(
                    name="case_embedding",
                    trainable=self._is_embedding_fine_tune,
                    shape=[self._case_vocab_manager.vocab_len-1, self._embedding_dim],
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(pretrained_case_word_embedding)
                )
            case_embedding = tf.concat(values=[pad_embeddings, case_embedding], axis=0)
            return case_embedding

    def _build_regex_embedding_layer(self):
        with tf.variable_scope("regex_embedding_layer"):
            regex_embedding = tf.get_variable(
                initializer=tf.truncated_normal(
                    [self._regex_vocab_manager.vocab_len - 1, self._embedding_dim],
                    stddev=0.5
                ),
                name='regex_embedding'
            )
            pad_embeddings = tf.get_variable(
                initializer=tf.zeros([1, self._embedding_dim]),
                name="regex_pad_embedding",
                trainable=False
            )
            regex_embedding = tf.concat(values=[pad_embeddings, regex_embedding], axis=0)
            return regex_embedding

    def _build_input_nodes(self, is_pretrained_embedding_used=False):
        with tf.name_scope('model_placeholder'):
            self._sentence_inputs = tf.placeholder(tf.int32, [self._actual_batch_size, self._max_sentence_length],
                                                   name="sentence_inputs")
            self._case_inputs = tf.placeholder(tf.int32, [self._actual_batch_size, self._max_case_length],
                                               name="case_inputs")
            self._sentence_length = tf.placeholder(tf.int32, [self._actual_batch_size], name="sentence_length")
            self._case_length = tf.placeholder(tf.int32, [self._actual_batch_size], name="case_length")
            self._sentence_masks = tf.placeholder(tf.float32, [self._actual_batch_size, self._max_sentence_length],
                                                  name="sentence_masks")
            self._case_masks = tf.placeholder(tf.float32, [self._actual_batch_size, self._max_case_length],
                                              name="case_masks")
            self._rnn_output_keep_prob = tf.placeholder(tf.float32, name="output_keep_prob")
            self._rnn_input_keep_prob = tf.placeholder(tf.float32, name="input_keep_prob")

            if not self._is_test:
                self._regex_inputs = tf.placeholder(tf.int32, [self._actual_batch_size, self._max_regex_length],
                                                    name="regex_inputs")
                self._regex_targets = tf.placeholder(tf.int32, [self._batch_size, self._max_regex_length],
                                                     name="regex_targets")
                self._regex_masks = tf.placeholder(tf.float32, [self._batch_size, self._max_regex_length],
                                                   name="regex_targets")

    def _build_sentence_rnn(self, sentence_embedded, sequence_length):
        with tf.variable_scope("sentence_encoder"):
            with tf.variable_scope("cell"):
                sentence_cell = LSTMCell(
                    num_units=self._hidden_dim,
                    state_is_tuple=True
                )
                sentence_cell = tf.contrib.rnn.DropoutWrapper(
                    cell=sentence_cell,
                    input_keep_prob=self._rnn_input_keep_prob,
                    output_keep_prob=self._rnn_output_keep_prob
                )
                sentence_cell = tf.contrib.rnn.MultiRNNCell(
                    [sentence_cell] * self._layers,
                    state_is_tuple=True
                )
            with tf.variable_scope("encode"):
                sentence_encoder_outputs, sentence_encoder_states = tf.nn.dynamic_rnn(
                    cell=sentence_cell,
                    inputs=sentence_embedded,
                    sequence_length=sequence_length,
                    dtype=tf.float32
                )
            return sentence_encoder_outputs, sentence_encoder_states

    def _build_case_rnn(self, case_embedded, sequence_length, init_states):
        with tf.variable_scope("case_encoder"):
            with tf.variable_scope("cell"):
                case_cell = LSTMCell(
                    num_units=self._hidden_dim,
                    state_is_tuple=True
                )
                case_cell = tf.contrib.rnn.DropoutWrapper(
                    cell=case_cell,
                    input_keep_prob=self._rnn_input_keep_prob,
                    output_keep_prob=self._rnn_output_keep_prob
                )
                case_cell = tf.contrib.rnn.MultiRNNCell(
                    [case_cell] * self._layers,
                    state_is_tuple=True
                )
            with tf.variable_scope("encode"):
                case_encoder_outputs, case_encoder_states = tf.nn.dynamic_rnn(
                    cell=case_cell,
                    inputs=case_embedded,
                    sequence_length=sequence_length,
                    initial_state=init_states,
                    dtype=tf.float32
                )
            return case_encoder_outputs, case_encoder_states

    def _encode(self):
        """
        Encode sentence and test case
        :return:
        """
        sentence_embedding = self._build_sentence_embedding_layer(self._pretrained_sentence_embedding)
        case_embedding = self._build_case_embedding_layer(self._pretrained_case_embedding)

        sentence_embedded = tf.nn.embedding_lookup(sentence_embedding, self._sentence_inputs)
        case_embedded = tf.nn.embedding_lookup(case_embedding, self._case_inputs)

        sentence_encoder_outputs, sentence_encoder_states = self._build_sentence_rnn(
            sentence_embedded=sentence_embedded,
            sequence_length=self._sentence_length
        )
        case_encoder_outputs, case_encoder_states = self._build_case_rnn(
            case_embedded=case_embedded,
            sequence_length=self._case_length,
            init_states=sentence_encoder_states
        )

        return sentence_encoder_outputs, sentence_encoder_states, case_encoder_outputs, case_encoder_states

    def _build_decoder_cell(self):
        """
        Build Decoder Cell
        :return:
        """
        with tf.variable_scope("cell"):
            regex_cell = LSTMCell(
                num_units=self._hidden_dim,
                state_is_tuple=True
            )
            regex_cell = tf.contrib.rnn.DropoutWrapper(
                cell=regex_cell,
                input_keep_prob=self._rnn_input_keep_prob,
                output_keep_prob=self._rnn_output_keep_prob
            )
            regex_cell = tf.contrib.rnn.MultiRNNCell(
                [regex_cell] * self._layers,
                state_is_tuple=True
            )
            return regex_cell

    def _build_attention_parameters(self):
        with tf.variable_scope("regex_sentence_attention"):
            rs_weight_a = tf.get_variable(
                initializer=tf.truncated_normal([self._hidden_dim, self._hidden_dim], dtype=tf.float32),
                name="weight_a"
            )
            rs_weight_c = tf.get_variable(
                initializer=tf.truncated_normal([self._hidden_dim, self._hidden_dim * 2], dtype=tf.float32),
                name="weight_c"
            )

        with tf.variable_scope("regex_case_attention"):
            rc_weight_a = tf.get_variable(
                initializer=tf.truncated_normal([self._hidden_dim * 2, self._hidden_dim], dtype=tf.float32, stddev=0.5),
                name="weight_a"
            )
            rc_weight_c = tf.get_variable(
                initializer=tf.truncated_normal([self._hidden_dim, self._hidden_dim * 2], dtype=tf.float32, stddev=0.5),
                name="weight_c"
            )

        return rs_weight_a, rs_weight_c, rc_weight_a, rc_weight_c

    def _calc_regex_sentence_attention(self, weight_a, weight_c, hs, sentence_outputs, sentence_masks):
        """
        Calculate regex and sentence attention
        :param weight_a:            [hidden_dim, hidden_dim],
        :param weight_c:            [hidden_dim, hidden_dim*2]
        :param hs:                  [batch_size, hidden_dim]
        :param sentence_outputs:    [batch_size, max_sentence_len, hidden_dim]
        :param sentence_masks:      [batch_size, max_sentence_len]
        :return:
            [batch_size, hidden_dim]
        """

        with tf.name_scope("calc_regex_sentence_attention"):
            # [batch_size*hidden_dim, 1]
            _scores = tf.reshape(
                tf.matmul(hs, weight_a),
                shape=[self._actual_batch_size * self._hidden_dim, 1]
            )
            # [batch_size*hidden_dim, max_sentence_length]
            transposed_sentence_outputs = tf.reshape(
                tf.transpose(sentence_outputs, perm=[0, 2, 1]),
                shape=[self._actual_batch_size * self._hidden_dim, self._max_sentence_length]
            )
            # [batch_size, max_sentence_length]
            scores = tf.reduce_sum(
                tf.transpose(
                    tf.reshape(
                        tf.multiply(
                            transposed_sentence_outputs, _scores
                        ),
                        shape=[self._actual_batch_size, self._hidden_dim, self._max_sentence_length]
                    ),
                    perm=[0, 2, 1]
                ),
                axis=2
            )
            # softmax, [batch_size, max_sentence_length]
            softmax_weights = softmax_with_mask(scores, sentence_masks)
            # [batch_size, max_sentence_length]
            reshaped_weights = tf.reshape(
                softmax_weights, shape=[self._actual_batch_size, self._max_sentence_length, 1]
            )
            # context_vector, [batch_size, hidden_dim]
            context_vector = tf.reduce_sum(
                tf.multiply(reshaped_weights, sentence_outputs),
                axis=1
            )
            # [batch_size, hidden_dim*2]
            concatenated = tf.concat((context_vector, tf.reshape(hs, shape=[-1, self._hidden_dim])), axis=1)

            # [batch_size, hidden_dim]
            attentive_hs = tf.tanh(
                tf.transpose(
                    tf.matmul(
                        weight_c,
                        concatenated,
                        transpose_b=True
                    ),
                    perm=[1, 0]
                )
            )
            return attentive_hs

    def _calc_regex_case_attention(self, weight_a, weight_c, hs, sentence_attentive_hs, case_outputs, case_masks):
        """
        Concatenate hs and sentence_attentive_hs
        :param weight_a:                [hidden_dim*2, hidden_dim]
        :param weight_c:                [hidden_dim, hidden_dim*2]
        :param hs:                      [batch_size, hidden_dim]
        :param sentence_attentive_hs:   [batch_size, hidden_dim]
        :param case_outputs:            [batch_size, max_case_length, hidden_dim]
        :param case_masks:              [batch_size, max_case_length]
        :return:
        """
        with tf.name_scope("calc_regex_case_attention"):
            # [batch_size, hidden_dim*2]
            _hs = tf.concat(
                (
                    tf.reshape(hs, shape=[-1, self._hidden_dim]),
                    sentence_attentive_hs
                ),
                axis=1
            )
            # [batch_size*hidden_dim, 1]
            _scores = tf.reshape(
                tf.matmul(_hs, weight_a),
                shape=[self._actual_batch_size * self._hidden_dim, 1]
            )
            # [batch_size*hidden_dim, max_case_length]
            transposed_sentence_outputs = tf.reshape(
                tf.transpose(case_outputs, perm=[0, 2, 1]),
                shape=[self._actual_batch_size * self._hidden_dim, self._max_case_length]
            )
            # [batch_size, max_case_length]
            scores = tf.reduce_sum(
                tf.transpose(
                    tf.reshape(
                        tf.multiply(
                            transposed_sentence_outputs, _scores
                        ),
                        shape=[self._actual_batch_size, self._hidden_dim, self._max_case_length]
                    ),
                    perm=[0, 2, 1]
                ),
                axis=2
            )
            # softmax, [batch_size, max_case_length]
            softmax_weights = softmax_with_mask(scores, case_masks)
            # [batch_size, max_case_length]
            reshaped_weights = tf.reshape(
                softmax_weights, shape=[self._actual_batch_size, self._max_case_length, 1]
            )
            # context_vector, [batch_size, hidden_dim]
            context_vector = tf.reduce_sum(
                tf.multiply(reshaped_weights, case_outputs),
                axis=1
            )
            # [batch_size, hidden_dim*2]
            concatenated = tf.concat((context_vector, tf.reshape(hs, shape=[-1, self._hidden_dim])), axis=1)
            # [batch_size, hidden_dim]
            attentive_hs = tf.tanh(
                tf.transpose(
                    tf.matmul(
                        weight_c,
                        concatenated,
                        transpose_b=True
                    ),
                    perm=[1, 0]
                )
            )
            return attentive_hs

    def _train_decode(self, sentence_outputs, case_outputs, encoder_states, encoder_hidden_states):
        """
        Build Decoder
        :param sentence_outputs:
        :param case_outputs:
        :param encoder_states:
        :param encoder_hidden_states:
        :return:
            max_pooling_result: [batch_size, hidden_dim]
        """
        assert not self._is_test
        regex_embedding = self._build_regex_embedding_layer()
        regex_embedded = tf.nn.embedding_lookup(regex_embedding, self._regex_inputs)

        with tf.variable_scope("regex_decoder"):
            regex_cell = self._build_decoder_cell()

        rs_weight_a, rs_weight_c, rc_weight_a, rc_weight_c = self._build_attention_parameters()

        with tf.name_scope("train_decode"):
            def __cond(curr_ts, last_decoder_states, last_hs, decoder_outputs_array):
                return tf.less(curr_ts, self._max_regex_length)

            def __loop_body(curr_ts, _last_decoder_states, last_hs, decoder_outputs_array):
                """
                Run Decode step by step
                :param curr_ts:               Scalar
                :param _last_decoder_states:   LSTMStateTuple
                :param last_hs:               [batch_size, hidden_dim]
                :param decoder_outputs_array: TensorArray
                :return:
                """

                inputs = tf.slice(
                    regex_embedded,
                    begin=[0, curr_ts, 0],
                    size=[self._actual_batch_size, 1, self._embedding_dim]
                )

                sentence_attentive_vector = self._calc_regex_sentence_attention(
                    hs=last_hs,
                    sentence_outputs=sentence_outputs,
                    sentence_masks=self._sentence_masks,
                    weight_c=rs_weight_c,
                    weight_a=rs_weight_a
                )

                case_attentive_vector = self._calc_regex_case_attention(
                    hs=last_hs,
                    sentence_attentive_hs=sentence_attentive_vector,
                    case_outputs=case_outputs,
                    case_masks=self._case_masks,
                    weight_c=rc_weight_c,
                    weight_a=rc_weight_a
                )

                _inputs = tf.concat(
                    (
                        inputs,
                        tf.reshape(
                            sentence_attentive_vector,
                            shape=[self._actual_batch_size, 1, self._hidden_dim]
                        ),
                        tf.reshape(
                            case_attentive_vector,
                            shape=[self._actual_batch_size, 1, self._hidden_dim]
                        )
                    ),
                    axis=2
                )

                _decoder_outputs, _decoder_states = tf.nn.dynamic_rnn(
                    cell=regex_cell,
                    inputs=_inputs,
                    dtype=tf.float32,
                    initial_state=_last_decoder_states,
                )
                decoder_outputs_array = decoder_outputs_array.write(
                    curr_ts,
                    tf.reshape(
                        _decoder_outputs,
                        shape=[self._actual_batch_size, self._hidden_dim]
                    )
                )

                next_ts = tf.add(curr_ts, 1)

                reshaped_decoder_outputs = tf.reshape(_decoder_outputs,
                                                      shape=[self._actual_batch_size, self._hidden_dim])

                return next_ts, _decoder_states, reshaped_decoder_outputs, decoder_outputs_array

            total_ts, last_decoder_states, last_hidden_states, decoder_outputs = tf.while_loop(
                body=__loop_body,
                cond=__cond,
                loop_vars=[
                    tf.constant(0),
                    encoder_states,
                    encoder_hidden_states,
                    tf.TensorArray(dtype=tf.float32, size=self._max_regex_length),
                ]
            )

            # [batch_size, max_regex_length, hidden_dim]
            decoder_outputs = tf.transpose(
                decoder_outputs.stack(name="regex_outputs"),
                perm=[1, 0, 2]
            )

        with tf.variable_scope("max_pooling"):
            pooling_weight = tf.get_variable(
                initializer=tf.truncated_normal([self._hidden_dim, self._hidden_dim], stddev=0.5),
                name="weight"
            )

        with tf.name_scope("calc_train_max_pooling"):
            # [batch_size, max_regex_length, hidden_dim]
            weighted_outputs = tf.reshape(
                tf.transpose(
                    tf.tanh(
                        tf.matmul(
                            pooling_weight,
                            tf.reshape(
                                decoder_outputs,
                                shape=[self._actual_batch_size * self._max_regex_length, self._hidden_dim]
                            ),
                            transpose_b=True
                        )
                    ),
                    perm=[1, 0]
                ),
                shape=[self._actual_batch_size, self._max_regex_length, self._hidden_dim]
            )

            # [batch_size, max_regex_length, hidden_dim, case_num]
            divided_outputs = tf.transpose(
                tf.reshape(
                    weighted_outputs,
                    shape=[self._batch_size, self._case_num, self._max_regex_length, self._hidden_dim]
                ),
                perm=[0, 2, 3, 1]
            )

            # [batch_size, max_regex_length, hidden_dim]
            max_pooling_result = tf.reduce_max(
                divided_outputs,
                axis=3
            )
            return max_pooling_result

    def _test_decode(self, sentence_outputs, case_outputs, encoder_states, encoder_hidden_states):
        """
        Build Test
        Test Batch Size: 1
        :param sentence_outputs:        [1*case_num, max_sentence_len, hidden_dim]
        :param case_outputs:            [1*case_num, max_case_len, hidden_dim]
        :param encoder_states:          LSTMStateTuple
        :param encoder_hidden_states:   [1*case_num, hidden_dim]
        :return:
        """
        assert self._is_test
        regex_embedding = self._build_regex_embedding_layer()

        with tf.variable_scope("regex_decoder"):
            regex_cell = self._build_decoder_cell()

        with tf.variable_scope("max_pooling"):
            pooling_weight = tf.get_variable(
                initializer=tf.truncated_normal([self._hidden_dim, self._hidden_dim], stddev=0.5),
                name="weight"
            )

        with tf.variable_scope("prediction_softmax"):
            softmax_weights = tf.get_variable(
                initializer=tf.truncated_normal(
                    [self._regex_vocab_manager.vocab_len, self._hidden_dim],
                    stddev=0.5
                ),
                name="prediction_softmax_weight"
            )

        rs_weight_a, rs_weight_c, rc_weight_a, rc_weight_c = self._build_attention_parameters()

        with tf.name_scope("test_decode"):
            def __cond(curr_ts, last_decoder_states, last_hs, last_prediction, decoder_outputs_array, prediction_array):
                return tf.less(curr_ts, self._max_regex_length)

            def __loop_body(curr_ts, last_decoder_states, last_hs, last_prediction, decoder_outputs_array,
                            prediction_array):
                """
                Run Decode step by step
                :param curr_ts:               Scalar
                :param last_decoder_states:   LSTMStateTuple
                :param last_hs:               [case_num, hidden_dim]
                :param last_prediction:       [case_num], index
                :param decoder_outputs_array: TensorArray
                :param prediction_array:      TensorArray
                :return:
                """

                inputs = tf.nn.embedding_lookup(
                    params=regex_embedding,
                    ids=tf.reshape(
                        last_prediction,
                        shape=[self._case_num, 1]
                    )
                )

                sentence_attentive_vector = self._calc_regex_sentence_attention(
                    hs=last_hs,
                    sentence_outputs=sentence_outputs,
                    sentence_masks=self._sentence_masks,
                    weight_c=rs_weight_c,
                    weight_a=rs_weight_a
                )

                case_attentive_vector = self._calc_regex_case_attention(
                    hs=last_hs,
                    sentence_attentive_hs=sentence_attentive_vector,
                    case_outputs=case_outputs,
                    case_masks=self._case_masks,
                    weight_c=rc_weight_c,
                    weight_a=rc_weight_a
                )

                _inputs = tf.concat(
                    (
                        inputs,
                        tf.reshape(
                            sentence_attentive_vector,
                            shape=[self._actual_batch_size, 1, self._hidden_dim]
                        ),
                        tf.reshape(
                            case_attentive_vector,
                            shape=[self._actual_batch_size, 1, self._hidden_dim]
                        )
                    ),
                    axis=2
                )

                _decoder_outputs, _decoder_states = tf.nn.dynamic_rnn(
                    cell=regex_cell,
                    inputs=_inputs,
                    dtype=tf.float32,
                    initial_state=last_decoder_states,
                )

                next_ts = tf.add(curr_ts, 1)

                # Max pooling
                # [hidden_dim, case_num]
                weighted_outputs = tf.tanh(
                    tf.matmul(
                        pooling_weight,
                        tf.reshape(
                            _decoder_outputs,
                            shape=[self._case_num, self._hidden_dim]
                        ),
                        transpose_b=True
                    )
                )

                # [hidden_dim]
                max_pooling_result = tf.reduce_max(
                    weighted_outputs,
                    axis=1
                )

                decoder_outputs_array = decoder_outputs_array.write(curr_ts, max_pooling_result)

                _weighted = tf.reshape(
                    tf.matmul(
                        softmax_weights,
                        tf.reshape(max_pooling_result,
                                   shape=[self._hidden_dim, 1])
                    ),
                    shape=[self._regex_vocab_manager.vocab_len]
                )
                softmax_outputs = tf.nn.softmax(_weighted)

                curr_prediction = tf.cast(tf.arg_max(softmax_outputs, dimension=0), dtype=tf.int32)

                prediction_array = prediction_array.write(curr_ts, curr_prediction)

                copy_prediction = tf.multiply(
                    tf.ones([self._case_num], dtype=tf.int32),
                    curr_prediction
                )

                reshaped_decoder_outputs = tf.reshape(_decoder_outputs,
                                                      shape=[self._case_num, self._hidden_dim])

                return next_ts, _decoder_states, reshaped_decoder_outputs, copy_prediction, decoder_outputs_array, prediction_array

            total_ts, last_decoder_states, last_hidden_states, prediction, decoder_outputs, all_predictions = tf.while_loop(
                body=__loop_body,
                cond=__cond,
                loop_vars=[
                    tf.constant(0),
                    encoder_states,
                    encoder_hidden_states,
                    tf.constant([self._regex_vocab_manager.GO_TOKEN_ID] * self._case_num, dtype=tf.int32),
                    tf.TensorArray(dtype=tf.float32, size=self._max_regex_length),
                    tf.TensorArray(dtype=tf.int32, size=self._max_regex_length)
                ]
            )
            # [max_regex_length]
            prediction_tensor = all_predictions.stack(name="regex_predictions")

            return prediction_tensor

    def _build_train_graph(self):
        """
        Build Train Graph
        :return:
        """
        self._build_input_nodes()

        sentence_encoder_outputs, sentence_encoder_states, case_encoder_outputs, case_encoder_states = self._encode()

        def get_last_relevant(output, length):
            slices = list()
            for idx, l in enumerate(tf.unstack(length)):
                last = tf.slice(output, begin=[idx, l - 1, 0], size=[1, 1, self._hidden_dim])
                slices.append(last)
            lasts = tf.concat(slices, 0)
            return lasts

        sentence_last_outputs = tf.reshape(
            get_last_relevant(sentence_encoder_outputs, self._sentence_length),
            shape=[-1, self._hidden_dim]
        )
        case_last_outputs = tf.reshape(
            get_last_relevant(case_encoder_outputs, self._case_length),
            shape=[-1, self._hidden_dim]
        )

        # [batch_size, max_regex_length, hidden_dim]
        max_pooling_result = self._train_decode(
            sentence_outputs=sentence_encoder_outputs,
            case_outputs=case_encoder_outputs,
            encoder_states=case_encoder_states,
            encoder_hidden_states=case_last_outputs
        )

        with tf.variable_scope("prediction_softmax"):
            softmax_weights = tf.get_variable(
                initializer=tf.truncated_normal(
                    [self._regex_vocab_manager.vocab_len, self._hidden_dim],
                    stddev=0.5
                ),
                name="prediction_softmax_weight"
            )

        with tf.name_scope("calc_prediction_softmax"):
            _weighted = tf.reshape(
                tf.transpose(
                    tf.matmul(
                        softmax_weights,
                        tf.reshape(max_pooling_result, shape=[self._batch_size * self._max_regex_length, self._hidden_dim]),
                        transpose_b=True
                    ),
                    perm=[1, 0]
                ),
                shape=[self._batch_size, self._max_regex_length, self._regex_vocab_manager.vocab_len]
            )
            softmax_outputs = tf.nn.softmax(_weighted)

            self._predictions = tf.arg_max(softmax_outputs, dimension=2)

        # training, define loss
        with tf.name_scope("loss"):
            self._loss = tf.contrib.seq2seq.sequence_loss(
                logits=_weighted,
                targets=self._regex_targets,
                weights=self._regex_masks
            )

        with tf.name_scope('back_propagation'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)

            # clipped at 5 to alleviate the exploding gradient problem
            self._gvs = optimizer.compute_gradients(self._loss)
            self._capped_gvs = [(tf.clip_by_value(grad, -self._gradient_clip, self._gradient_clip), var) for grad, var
                                in self._gvs]
            self._optimizer = optimizer.apply_gradients(self._capped_gvs)

    def _build_test_graph(self):
        self._build_input_nodes()

        sentence_encoder_outputs, sentence_encoder_states, case_encoder_outputs, case_encoder_states = self._encode()

        def get_last_relevant(output, length):
            slices = list()
            for idx, l in enumerate(tf.unstack(length)):
                last = tf.slice(output, begin=[idx, l - 1, 0], size=[1, 1, self._hidden_dim])
                slices.append(last)
            lasts = tf.concat(slices, 0)
            return lasts

        sentence_last_outputs = tf.reshape(
            get_last_relevant(sentence_encoder_outputs, self._sentence_length),
            shape=[-1, self._hidden_dim]
        )
        case_last_outputs = tf.reshape(
            get_last_relevant(case_encoder_outputs, self._case_length),
            shape=[-1, self._hidden_dim]
        )

        predictions = self._test_decode(
            sentence_outputs=sentence_encoder_outputs,
            case_outputs=case_encoder_outputs,
            encoder_states=case_encoder_states,
            encoder_hidden_states=case_last_outputs
        )

        self._predictions = predictions

    def _build_train_feed(self, batch):
        feed_dict = dict()
        feed_dict[self._sentence_inputs] = batch.sentences
        feed_dict[self._sentence_length] = batch.sentence_length
        feed_dict[self._case_inputs] = batch.cases
        feed_dict[self._case_length] = batch.case_length
        feed_dict[self._sentence_masks] = batch.sentence_masks
        feed_dict[self._case_masks] = batch.case_masks
        feed_dict[self._regex_inputs] = batch.regexs
        feed_dict[self._regex_targets] = batch.regex_targets
        feed_dict[self._regex_masks] = batch.regex_masks
        feed_dict[self._rnn_input_keep_prob] = 1. - self._dropout
        feed_dict[self._rnn_output_keep_prob] = 1. - self._dropout
        return feed_dict

    def _build_test_feed(self, batch):
        feed_dict = dict()
        feed_dict[self._sentence_inputs] = batch.sentences
        feed_dict[self._sentence_length] = batch.sentence_length
        feed_dict[self._case_inputs] = batch.cases
        feed_dict[self._case_length] = batch.case_length
        feed_dict[self._sentence_masks] = batch.sentence_masks
        feed_dict[self._case_masks] = batch.case_masks
        feed_dict[self._rnn_input_keep_prob] = 1.
        feed_dict[self._rnn_output_keep_prob] = 1.
        return feed_dict

    def train(self, batch):
        assert not self._is_test
        feed_dict = self._build_train_feed(batch)
        return self._predictions, self._loss, self._optimizer, feed_dict

    def predict(self, batch):
        feed_dict = self._build_test_feed(batch)
        return self._predictions, feed_dict
