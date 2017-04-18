# coding=utf8

"""
Classification model
"""

import util
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell


class Model:

    FULLY_CONNECTED_LAYER_OUTPUT = 2

    def __init__(self, sentence_vocab_manager, case_vocab_manager, opts, is_test=False):
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

    def _build_case_rnn(self, case_embedded, sequence_length):
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
            sequence_length=self._case_length
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

        last_outputs_concat = tf.concat(
            values=[sentence_last_outputs, case_last_outputs],
            axis=1
        )

        # shape: [batch_size, FULLY_CONNECTED_LAYER_OUTPUT]
        outputs = self._build_fully_connected_layer(last_outputs_concat)

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

