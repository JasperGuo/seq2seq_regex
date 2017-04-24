# coding=utf8

# coding=utf8

"""
Conditional Classification model, encode the test case conditioned on the sentence
"""
import sys
sys.path += [".."]
import util
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

    FULLY_CONNECTED_LAYER_OUTPUT = 2

    def __init__(self, sentence_vocab_manager, case_vocab_manager, opts, attention=None, is_test=False):
        self._sentence_vocab_manager = sentence_vocab_manager
        self._case_vocab_manager = case_vocab_manager

        self._hidden_dim = util.get_value(opts, "hidden_dim", 150)
        self._layers = util.get_value(opts, "layer", 2)
        self._max_sentence_length = util.get_value(opts, "max_sentence_length", 30)
        self._max_case_length = util.get_value(opts, "max_case_length", 100)
        self._embedding_dim = util.get_value(opts, "embedding_dim", 150)
        self._learning_rate = util.get_value(opts, "learning_rate", 0.01)
        self._gradient_clip = util.get_value(opts, "gradient_clip", 5)
        self._dropout = util.get_value(opts, "dropout", 0.25)

        self._batch_size = util.get_value(opts, "batch_size", 5)

        self._is_test = is_test
        self._attention = attention
        self._build_graph()

    def _build_sentence_embedding_layer(self):
        with tf.variable_scope("sentence_embedding_layer"):
            sentence_embedding = tf.get_variable(
                initializer=tf.truncated_normal(
                    [self._sentence_vocab_manager.vocab_len - 1, self._embedding_dim],
                    stddev=0.5
                ),
                name='sentence_embedding'
            )
            pad_embeddings = tf.get_variable(
                initializer=tf.zeros([1, self._embedding_dim]),
                name="sentence_pad_embedding",
                trainable=False
            )
            sentence_embedding = tf.concat(values=[pad_embeddings, sentence_embedding], axis=0)
            return sentence_embedding

    def _build_case_embedding_layer(self):
        with tf.variable_scope("case_embedding_layer"):
            case_embedding = tf.get_variable(
                initializer=tf.truncated_normal(
                    [self._case_vocab_manager.vocab_len - 1, self._embedding_dim],
                    stddev=0.5
                ),
                name='case_embedding'
            )
            pad_embeddings = tf.get_variable(
                initializer=tf.zeros([1, self._embedding_dim]),
                name="case_pad_embedding",
                trainable=False
            )
            case_embedding = tf.concat(values=[pad_embeddings, case_embedding], axis=0)
            return case_embedding

    def _build_input_nodes(self):
        with tf.name_scope('model_placeholder'):
            self._sentence_inputs = tf.placeholder(tf.int32, [self._batch_size, self._max_sentence_length], name="sentence_inputs")
            self._case_inputs = tf.placeholder(tf.int32, [self._batch_size, self._max_case_length], name="case_inputs")
            self._sentence_length = tf.placeholder(tf.int32, [self._batch_size], name="source_length")
            self._case_length = tf.placeholder(tf.int32, [self._batch_size], name="case_length")
            self._sentence_masks = tf.placeholder(tf.float32, [self._batch_size, self._max_sentence_length], name="sentence_mask")
            self._case_masks = tf.placeholder(tf.float32, [self._batch_size, self._max_case_length], name="case_mask")
            self._rnn_output_keep_prob = tf.placeholder(tf.float32, name="output_keep_prob")
            self._rnn_input_keep_prob = tf.placeholder(tf.float32, name="input_keep_prob")
            self._labels = tf.placeholder(tf.int32, [self._batch_size], name="labels")

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

    def _build_fully_connected_layer(self, inputs):
        with tf.variable_scope("fully_connected"):
            outputs = tf.contrib.layers.fully_connected(
                inputs=inputs,
                num_outputs=self.FULLY_CONNECTED_LAYER_OUTPUT,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer()
            )
            return outputs

    def _build_sentence_level_attention(self, sentence_encode_outputs, case_last_encode_outputs, sentence_masks):
        """
        Build Sentence Level Attention Layer
        :param sentence_encode_outputs:
        :param case_last_encode_outputs: [batch_size, hidden_dim]
        :param sentence_masks:           [batch_size, max_sentence_length]
        :return:
        """
        with tf.variable_scope("sentence_level_attention"):
            weight_y = tf.get_variable(
                initializer=tf.truncated_normal(
                    [self._hidden_dim, self._hidden_dim],
                    stddev=0.5
                ),
                name="weight_y"
            )
            weight_h = tf.get_variable(
                initializer=tf.truncated_normal(
                    [self._hidden_dim, self._hidden_dim],
                    stddev=0.5
                ),
                name="weight_h"
            )
            weight_ = tf.get_variable(
                initializer=tf.truncated_normal(
                    [self._hidden_dim, 1],
                    stddev=0.5
                ),
                name="weight_"
            )

        with tf.name_scope("calc_attention"):
            # Shape: [batch_size, hidden_dim, max_length]
            weighted_sentence_encode_outputs = tf.map_fn(
                lambda x: tf.matmul(weight_y, x, transpose_b=True),
                elems=sentence_encode_outputs
            )
            # Shape: [batch_size, hidden_dim, 1]
            weighted_case_last_outputs = tf.map_fn(
                lambda x: tf.matmul(weight_h, x),
                elems=tf.reshape(
                    case_last_encode_outputs,
                    shape=[self._batch_size, self._hidden_dim, 1]
                )
            )
            e_l = tf.ones(
                [
                    self._batch_size,
                    self._hidden_dim,
                    self._max_sentence_length
                ],
                dtype=tf.float32
            )
            # Shape: [batch_size, hidden_dim, max_length]
            outer_product = tf.multiply(weighted_case_last_outputs, e_l)
            M = tf.tanh(tf.add(weighted_sentence_encode_outputs, outer_product))

            # Shape: [batch_size, max_length, 1]
            weighted_M = tf.reshape(
                softmax_with_mask(
                    tf.reshape(
                        tf.map_fn(
                            lambda x: tf.matmul(weight_, x, transpose_a=True),
                            elems=M
                        ),
                        shape=[self._batch_size, self._max_sentence_length]
                    ),
                    sentence_masks
                ),
                shape=[self._batch_size, self._max_sentence_length, 1]
            )
            # Shape: [batch_size, hidden_dim]
            r = tf.reduce_sum(
                tf.multiply(sentence_encode_outputs, weighted_M),
                axis=1
            )

        with tf.variable_scope("non_linearity_transformation_weight") as f:
            weight_p = tf.get_variable(
                initializer=tf.truncated_normal(
                    [self._hidden_dim, self._hidden_dim],
                    stddev=0.5
                ),
                name="weight_p"
            )
            weight_x = tf.get_variable(
                initializer=tf.truncated_normal(
                    [self._hidden_dim, self._hidden_dim],
                    stddev=0.5
                ),
                name="weight_x"
            )

        with tf.name_scope("non_linearity_transform"):
            outputs = tf.tanh(
                tf.add(
                    tf.map_fn(
                        lambda x: tf.matmul(weight_p, x),
                        elems=tf.reshape(
                            r,
                            shape=[self._batch_size, self._hidden_dim, 1]
                        )
                    ),
                    tf.map_fn(
                        lambda x: tf.matmul(weight_x, x),
                        elems=tf.reshape(
                            case_last_encode_outputs,
                            shape=[self._batch_size, self._hidden_dim, 1]
                        )
                    ),
                )
            )

        return tf.reshape(
            outputs,
            shape=[self._batch_size, self._hidden_dim]
        ), weighted_M

    def _build_word_level_attention(self, sentence_encode_outputs, case_encode_outputs, case_last_encode_outputs, case_length, sentence_masks):
        """
        Build Word by Word Attention
        :param sentence_encode_outputs:
        :param case_encode_outputs:
        :param case_last_encode_outputs:
        :param case_length:
               attention_outputs: [batch_size, hidden_dim],
               attention_weights: [batch_size, max_case_length, max_sentence_length]
        :return:
        """
        with tf.variable_scope("sentence_level_attention"):
            weight_y = tf.get_variable(
                initializer=tf.truncated_normal(
                    [self._hidden_dim, self._hidden_dim],
                    stddev=0.5
                ),
                name="weight_y"
            )
            weight_h = tf.get_variable(
                initializer=tf.truncated_normal(
                    [self._hidden_dim, self._hidden_dim],
                    stddev=0.5
                ),
                name="weight_h"
            )
            weight_r = tf.get_variable(
                initializer=tf.truncated_normal(
                    [self._hidden_dim, self._hidden_dim],
                    stddev=0.5
                ),
                name="weight_r"
            )
            weight_t = tf.get_variable(
                initializer=tf.truncated_normal(
                    [self._hidden_dim, self._hidden_dim],
                    stddev=0.5
                ),
                name="weight_t"
            )
            weight_ = tf.get_variable(
                initializer=tf.truncated_normal(
                    [self._hidden_dim, 1],
                    stddev=0.5
                ),
                name="weight_"
            )

        with tf.variable_scope("non_linearity_transformation_weight") as f:
            weight_p = tf.get_variable(
                initializer=tf.truncated_normal(
                    [self._hidden_dim, self._hidden_dim],
                    stddev=0.5
                ),
                name="weight_p"
            )
            weight_x = tf.get_variable(
                initializer=tf.truncated_normal(
                    [self._hidden_dim, self._hidden_dim],
                    stddev=0.5
                ),
                name="weight_x"
            )

        def __calc_attention(sentence_outputs, case_outputs, last_case_outputs, length, __sentence_mask):
            """
            :param sentence_outputs: [max_sentence_length, hidden_dim]
            :param case_outputs:     [length, hidden_dim]
            :param last_case_outputs [hidden_dim, 1]
            :param length:           Scalar
            :param __sentence_mask   Sentence mask, [1, max_sentence_length]
            :return:
                   outputs:     [1, hidden_dim]
                   weights:     [max_case_length, max_sentence_length]
            """

            def __update_weight_matrix(index, value, matrix):
                """
                Insert value to outputs, with respect to index
                :param index:   scalar
                :param value:   [max_sentence_length, 1]
                :param matrix,  [max_case_length, max_sentence_length]
                :return:        [max_case_length, max_sentence_length]
                """
                one_hot_matrix = tf.transpose(
                    tf.tile(
                        tf.expand_dims(
                            tf.cast(tf.equal(tf.range(self._max_case_length), index), dtype=tf.float32),
                            0
                        ),
                        [self._max_sentence_length, 1]
                    ),
                    perm=[1, 0]
                )
                ts_value = tf.multiply(
                    one_hot_matrix,
                    tf.reshape(
                        value,
                        shape=[1, self._max_sentence_length]
                    )
                )
                return tf.add(matrix, ts_value)

            def __cond(curr_ts, last_attentive_vector, se_outputs, ca_outputs, __weight_matrix):
                return tf.less(curr_ts, length)

            def __loop_body(curr_ts, last_attentive_vector, se_outputs, ca_outputs, __weight_matrix):
                """
                :param curr_ts:                 Scalar
                :param last_attentive_vector:   [hidden_dim, 1]
                :param se_outputs:              [max_length, hidden_dim]
                :param ca_outputs:              [length, hidden_dim]
                :return:
                """
                curr_case_outputs = tf.slice(
                    ca_outputs,
                    begin=[curr_ts, 0],
                    size=[1, self._hidden_dim]
                )
                # Shape: [hidden_dim, max_length]
                weighted_sentence_outputs = tf.matmul(weight_y, se_outputs, transpose_b=True)
                # Shape: [hidden_dim, 1]
                weighted_case_curr_ts_outputs = tf.add(
                    tf.matmul(weight_h, curr_case_outputs, transpose_b=True),
                    tf.matmul(weight_r, last_attentive_vector)
                )
                # Shape: [hidden_dim, max_length]
                e_l = tf.ones(
                    [
                        self._hidden_dim,
                        self._max_sentence_length
                    ],
                    dtype=tf.float32
                )
                # Shape: [hidden_dim, max_length]
                outer_product = tf.multiply(weighted_case_curr_ts_outputs, e_l)
                M = tf.tanh(tf.add(weighted_sentence_outputs, outer_product))

                # Shape: [max_length, 1]
                alpha = tf.reshape(
                    softmax_with_mask(
                        tf.matmul(
                            tf.reshape(weight_, shape=[1, self._hidden_dim]),
                            M
                        ),
                        __sentence_mask
                    ),
                    shape=[self._max_sentence_length, 1]
                )
                # Shape: [hidden_dim]
                r1 = tf.reduce_sum(
                    tf.multiply(se_outputs, alpha),
                    axis=0
                )
                r2 = tf.reshape(
                    tf.tanh(
                        tf.matmul(
                            weight_t,
                            last_attentive_vector
                        )
                    ),
                    shape=[self._hidden_dim]
                )
                r = tf.reshape(tf.add(r1, r2), shape=[self._hidden_dim, 1])

                __weight_matrix = __update_weight_matrix(curr_ts, alpha, __weight_matrix)

                return tf.add(curr_ts, 1), r, se_outputs, ca_outputs, __weight_matrix

            last_ts, last_attentive_r, __se_outputs, __ca_outputs, _weights = tf.while_loop(
                cond=__cond,
                body=__loop_body,
                loop_vars=[
                    tf.constant(0, dtype=tf.int32),
                    tf.zeros([self._hidden_dim, 1], dtype=tf.float32),
                    sentence_outputs,
                    case_outputs,
                    tf.zeros([self._max_case_length, self._max_sentence_length], dtype=tf.float32)
                ]
            )

            outputs = tf.reshape(
                tf.tanh(
                    tf.add(
                        tf.matmul(weight_p, last_attentive_r),
                        tf.matmul(weight_x, last_case_outputs)
                    )
                ),
                shape=[1, self._hidden_dim]
            )

            return outputs, _weights

        attention_outputs = list()
        attention_weights = list()
        for idx, l in enumerate(tf.unstack(case_length)):
            _sentence_outputs = tf.reshape(
                tf.slice(
                    sentence_encode_outputs,
                    begin=[idx, 0, 0],
                    size=[1, self._max_sentence_length, self._hidden_dim]
                ),
                shape=[self._max_sentence_length, self._hidden_dim]
            )
            _case_outputs = tf.reshape(
                tf.slice(
                    case_encode_outputs,
                    begin=[idx, 0, 0],
                    size=[1, l, self._hidden_dim]
                ),
                shape=[l, self._hidden_dim]
            )
            _case_last_outputs = tf.reshape(
                tf.slice(
                    case_last_encode_outputs,
                    begin=[idx, 0],
                    size=[1, self._hidden_dim]
                ),
                shape=[self._hidden_dim, 1]
            )
            _sentence_mask = tf.reshape(
                tf.slice(
                    sentence_masks,
                    begin=[idx, 0],
                    size=[1, self._max_sentence_length]
                ),
                shape=[1, self._max_sentence_length]
            )
            _outputs, _weights = __calc_attention(_sentence_outputs, _case_outputs, _case_last_outputs, l, _sentence_mask)
            attention_weights.append(
                tf.reshape(
                    _weights,
                    shape=[1, self._max_case_length, self._max_sentence_length]
                )
            )
            attention_outputs.append(_outputs)

        return tf.concat(attention_outputs, 0), tf.concat(attention_weights, 0)

    def _build_graph(self):
        self._build_input_nodes()
        sentence_embedding = self._build_sentence_embedding_layer()
        case_embedding = self._build_case_embedding_layer()

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

        if not self._attention:
            # shape: [batch_size, FULLY_CONNECTED_LAYER_OUTPUT]
            outputs = self._build_fully_connected_layer(case_last_outputs)
        elif self._attention == "sentence":
            # Sentence Level Attention
            _outputs, weights = self._build_sentence_level_attention(
                sentence_encode_outputs=sentence_encoder_outputs,
                case_last_encode_outputs=case_last_outputs,
                sentence_masks=self._sentence_masks
            )
            self._attention_weights = tf.reshape(
                weights,
                shape=[self._batch_size, self._max_sentence_length]
            )
            outputs = self._build_fully_connected_layer(_outputs)
        else:
            # Word by Word Attention
            _outputs, weights = self._build_word_level_attention(
                sentence_encode_outputs=sentence_encoder_outputs,
                case_encode_outputs=case_encoder_outputs,
                case_last_encode_outputs=case_last_outputs,
                case_length=self._case_length,
                sentence_masks=self._sentence_masks
            )
            self._attention_weights = weights
            outputs = self._build_fully_connected_layer(_outputs)

        self._predictions = tf.argmax(outputs, axis=1, name="predictions")

        if self._is_test:
            return

        # Calculate Loss
        with tf.name_scope("loss"):
            loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self._labels,
                logits=outputs
            )
            self._loss = tf.reduce_mean(loss_, name="loss")

        with tf.name_scope("backpropagation"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
            gvs = optimizer.compute_gradients(self._loss)
            capped_gvs = [(tf.clip_by_value(grad, -self._gradient_clip, self._gradient_clip), var) for grad, var in gvs]
            self._optimizer = optimizer.apply_gradients(capped_gvs)

    def _build_train_feed(self, batch):
        feed_dict = dict()
        feed_dict[self._sentence_inputs] = batch.sentences
        feed_dict[self._sentence_length] = batch.sentence_length
        feed_dict[self._case_inputs] = batch.cases
        feed_dict[self._case_length] = batch.case_length
        feed_dict[self._case_masks] = batch.case_masks
        feed_dict[self._sentence_masks] = batch.sentence_masks
        feed_dict[self._labels] = batch.labels
        feed_dict[self._rnn_input_keep_prob] = 1. - self._dropout
        feed_dict[self._rnn_output_keep_prob] = 1. - self._dropout
        return feed_dict

    def _build_test_feed(self, batch):
        feed_dict = dict()
        feed_dict[self._sentence_inputs] = batch.sentences
        feed_dict[self._sentence_length] = batch.sentence_length
        feed_dict[self._case_inputs] = batch.cases
        feed_dict[self._case_length] = batch.case_length
        feed_dict[self._case_masks] = batch.case_masks
        feed_dict[self._sentence_masks] = batch.sentence_masks
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

    def predict_with_weights(self, batch):
        assert self._attention
        feed_dict = self._build_test_feed(batch)
        return self._predictions, self._attention_weights, feed_dict

