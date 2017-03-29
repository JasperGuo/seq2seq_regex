# coding=utf8

import util
from data_provider.data_iterator import VocabManager
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
import numpy as np


class Model:
    def __init__(self, vocab_manager, opts, is_test):
        self._hidden_dim = util.get_value(opts, "hidden_dim", 128)
        self._dropout = util.get_value(opts, "dropout", 0.25)
        self._layers = util.get_value(opts, "layer", 1)
        self._max_length_encode = util.get_value(opts, "max_length_encode", 20)
        self._max_length_decode = util.get_value(opts, "max_length_decode", 20)
        self._vocab_manager = vocab_manager
        self._embedding_dim = util.get_value(opts, "embedding_dim", 128)
        self._learning_rate = util.get_value(opts, "learning_rate", 0.001)
        self._gradient_clip = util.get_value(opts, "gradient_clip", 5)

        self._batch_size = util.get_value(opts, "batch_size", 20)

        self._uniform_init_min = util.get_value(opts, "params_init_min", -0.08)
        self._uniform_init_max = util.get_value(opts, "params_init_max", 0.08)

        self._is_test = is_test

        if self._is_test:
            self._build_test_graph()
        else:
            self._build_train_graph()

    def _build_train_graph(self):
        self._build_encoder()
        self._build_train_decoder()

        with tf.variable_scope("attention"):
            self._attention_weight1 = tf.get_variable(
                initializer=tf.random_uniform(shape=[self._hidden_dim, self._hidden_dim],
                                              minval=self._uniform_init_min,
                                              maxval=self._uniform_init_max),
                name="weight_1"
            )
            self._attention_weight2 = tf.get_variable(
                initializer=tf.random_uniform(shape=[self._hidden_dim, self._hidden_dim],
                                              minval=self._uniform_init_min,
                                              maxval=self._uniform_init_max),
                name="weight_2"
            )

        with tf.variable_scope("softmax") as scope:
            self._softmax_weight = tf.get_variable(
                initializer=tf.truncated_normal([self._vocab_manager.vocab_target_len, self._hidden_dim], stddev=0.5),
                name="softmax_weight"
            )
            self._softmax_bias = tf.get_variable(
                initializer=tf.truncated_normal([self._vocab_manager.vocab_target_len], stddev=0.5),
                name="softmax_bias"
            )

        self._attention_decoder_outputs = self._calculate_attention(self._decoder_outputs, num=self._max_length_decode)

        (self._predictions, softmax_outputs) = self._predict(self._attention_decoder_outputs)

        # training, define loss
        with tf.name_scope("loss"):
            def prepare_index(index, x_shape, y_shape):
                # TODO: Transfer more efficiently
                """
                Shape index (batch_size * max_length_decode) to (batch_size * max_length_decode * 3)
                For tf.gather_ng to index the target word
                From:
                [
                    [0, 1, 2, 3],
                    [0, 1, 2, 3]
                ]
                to:
                [
                    [
                        [0, 0, 0], [0, 1, 1], [0, 1, 2], [0, 1, 3]
                    ],
                    [
                        [1, 0, 0], [1, 1, 1], [1, 1, 2], [1, 1, 3]
                    ]
                ]
                :param index:
                :return:
                """
                # shape: batch_size * max_length_decode
                first_idx_tail = tf.tile(tf.expand_dims(tf.range(tf.shape(index)[0]), dim=1), [1, tf.shape(index)[1]])
                # batch_size * max_length_decode
                reshape_second_idx_tail = tf.transpose(
                    tf.tile(tf.expand_dims(tf.range(tf.shape(index)[1]), dim=1), [1, tf.shape(index)[0]]))

                combine_second_first = tf.stack([reshape_second_idx_tail, first_idx_tail], axis=2)

                combine_second_index = tf.stack([reshape_second_idx_tail, index], axis=2)

                temp_result = tf.concat([combine_second_first, combine_second_index], axis=2)

                # shape: batch_size * max_length_decode * 3
                return tf.slice(temp_result, [0, 0, 1], [x_shape, y_shape, 3])

            self._labeled = tf.gather_nd(softmax_outputs,
                                         prepare_index(self._decoder_targets, tf.shape(softmax_outputs)[0],
                                                       tf.shape(softmax_outputs)[1]))
            self._predict_log_probability = tf.reduce_sum(
                tf.log(self._labeled),
                axis=1
            )

            self._loss = tf.negative(tf.reduce_mean(self._predict_log_probability))

            # tf.summary.scalar('loss', self._loss)

        with tf.name_scope('back_propagation'):
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self._learning_rate, decay=0.95)
            # self._optimizer = optimizer.minimize(self._loss)

            # clipped at 5 to alleviate the exploding gradient problem
            gvs = optimizer.compute_gradients(self._loss)
            capped_gvs = [(tf.clip_by_value(grad, -self._gradient_clip, self._gradient_clip), var) for grad, var in gvs]
            self._optimizer = optimizer.apply_gradients(capped_gvs)

    def _build_test_graph(self):
        """
        build test graph
        :return:
        """
        self._build_encoder()

        with tf.variable_scope("attention"):
            self._attention_weight1 = tf.get_variable(
                initializer=tf.random_uniform(shape=[self._hidden_dim, self._hidden_dim],
                                              minval=self._uniform_init_min,
                                              maxval=self._uniform_init_max),
                name="weight_1"
            )
            self._attention_weight2 = tf.get_variable(
                initializer=tf.random_uniform(shape=[self._hidden_dim, self._hidden_dim],
                                              minval=self._uniform_init_min,
                                              maxval=self._uniform_init_max),
                name="weight_2"
            )

        with tf.variable_scope("softmax") as scope:
            self._softmax_weight = tf.get_variable(
                initializer=tf.truncated_normal([self._vocab_manager.vocab_target_len, self._hidden_dim], stddev=0.5),
                name="softmax_weight"
            )
            self._softmax_bias = tf.get_variable(
                initializer=tf.truncated_normal([self._vocab_manager.vocab_target_len], stddev=0.5),
                name="softmax_bias"
            )

        with tf.name_scope('decoder_placeholder'):
            self._decoder_input_keep_prob = tf.placeholder(tf.float32, name="input_keep_prob")
            self._decoder_output_keep_prob = tf.placeholder(tf.float32, name="output_keep_prob")

        with tf.name_scope("decoder_cell"):
            decoder_cell = LSTMCell(num_units=self._hidden_dim, state_is_tuple=True)
            decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell,
                                                         input_keep_prob=self._decoder_input_keep_prob,
                                                         output_keep_prob=self._decoder_output_keep_prob)
            self._decoder_cell = tf.contrib.rnn.MultiRNNCell([decoder_cell] * self._layers,
                                                             state_is_tuple=True)

        # Target Vocab embedding
        with tf.name_scope('decoder_embedding'):
            # vocab_size - 1, manually add zero tensor for PADDING embeddings
            self._decoder_embeddings = tf.get_variable(
                name="decoder_embeddings",
                initializer=tf.truncated_normal([self._vocab_manager.vocab_target_len - 1, self._embedding_dim], stddev=0.5)
            )

            # zero embedding for <PAD>
            pad_embeddings = tf.get_variable(initializer=tf.zeros([1, self._embedding_dim]),
                                             name='decoder_pad_embedding',
                                             trainable=False)
            self._decoder_embeddings = tf.concat(values=[pad_embeddings, self._decoder_embeddings], axis=0)

        # decoder
        with tf.variable_scope('decoder'):
            def _loop_body(token_id, curr_ts, _predictions, states):
                decoder_embedded = tf.nn.embedding_lookup(self._decoder_embeddings, token_id)

                decoder_outputs, decoder_states = tf.nn.dynamic_rnn(self._decoder_cell,
                                                                    initial_state=states,
                                                                    inputs=decoder_embedded,
                                                                    dtype=tf.float32)
                attention_decoder_outputs = self._calculate_attention(decoder_outputs, num=1)
                prediction, softmax_outputs = self._predict(attention_decoder_outputs)
                prediction = tf.reshape(prediction, tf.shape(token_id))

                _predictions = _predictions.write(curr_ts, tf.gather_nd(prediction, [0]))
                return prediction, tf.add(curr_ts, 1), _predictions, decoder_states

            def _terminate_condition(token_id, curr_ts, predictions, states):
                """
                :return:
                """
                return tf.less(curr_ts, self._max_length_decode)

            self._last_prediction, time_steps, self._predictions, self._decoder_states = tf.while_loop(
                _terminate_condition,
                _loop_body,
                [
                    tf.constant([[VocabManager.GO_TOKEN_ID]], dtype=tf.int64),
                    tf.constant(0),
                    tf.TensorArray(dtype=tf.int64, size=self._max_length_decode),
                    self._encoder_states
                ]
            )
            self._predictions = self._predictions.stack()

    def _predict(self, outputs):
        """
        prediction
        :param outputs:
        :return:
        """
        # softmax output
        # weighted_output shape: batch_size * max_length_decoder * vocab_f_len
        self._weighted_outputs = tf.map_fn(
            lambda x: tf.add(tf.matmul(x, self._softmax_weight, transpose_b=True), self._softmax_bias),
            outputs)
        softmax_outpus = tf.nn.softmax(self._weighted_outputs)

        with tf.name_scope("prediction"):
            predictions = tf.arg_max(softmax_outpus, dimension=2)
        return predictions, softmax_outpus

    def _build_encoder(self):
        """
        Build Encoder
        :return:
        """
        # Encoder placeholder
        with tf.name_scope('encoder_placeholder'):
            # encoder inputs are integers, representing the index of words in the sentences
            self._encoder_inputs = tf.placeholder(tf.int32, [None, self._max_length_encode], name="inputs")
            self._encoder_output_keep_prob = tf.placeholder(tf.float32, name="output_keep_prob")
            self._encoder_input_keep_prob = tf.placeholder(tf.float32, name="input_keep_prob")

        # source vocab embedding
        with tf.name_scope('encoder_embedding'):
            # vocab_size - 1, manually add zero tensor for PADDING embeddings
            self._encoder_embeddings = tf.get_variable(
                initializer=tf.truncated_normal([self._vocab_manager.vocab_source_len - 1, self._embedding_dim], stddev=0.5),
                name="encoder_embeddings"
            )

            # zero embedding for <PAD>
            pad_embeddings = tf.get_variable(initializer=tf.zeros([1, self._embedding_dim]),
                                             name='encoder_pad_embedding', trainable=False)
            self._encoder_embeddings = tf.concat(values=[pad_embeddings, self._encoder_embeddings], axis=0)

            # replace vocab-id with embedding
            # self._embedded shape is self._batch_size * self._max_length_encode * self._embedding_dim
            self._encoder_embedded = tf.nn.embedding_lookup(self._encoder_embeddings, self._encoder_inputs)

        with tf.variable_scope('encoder'):
            # dropout is only available during training
            # define encoder cell
            with tf.name_scope("encoder_cell"):
                encoder_cell = LSTMCell(num_units=self._hidden_dim, state_is_tuple=True)
                encoder_cell = tf.contrib.rnn.DropoutWrapper(encoder_cell,
                                                             input_keep_prob=self._encoder_input_keep_prob,
                                                             output_keep_prob=self._encoder_output_keep_prob)
                encoder_cell = tf.contrib.rnn.MultiRNNCell([encoder_cell] * self._layers,
                                                           state_is_tuple=True)
            self._encoder_outputs, self._encoder_states = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                                            inputs=self._encoder_embedded,
                                                                            dtype=tf.float32)

    def _build_train_decoder(self):
        """
        Build train phase decoder
        :return:
        """
        with tf.name_scope('decoder_placeholder'):
            self._decoder_input_keep_prob = tf.placeholder(tf.float32, name="input_keep_prob")
            self._decoder_output_keep_prob = tf.placeholder(tf.float32, name="output_keep_prob")
            self._decoder_inputs = tf.placeholder(tf.int32, [None, None], name="inputs")
            self._decoder_targets = tf.placeholder(tf.int32, [None, None], name="targets")

        with tf.name_scope("decoder_cell"):
            decoder_cell = LSTMCell(num_units=self._hidden_dim, state_is_tuple=True)
            decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell,
                                                         input_keep_prob=self._decoder_input_keep_prob,
                                                         output_keep_prob=self._decoder_output_keep_prob)
            self._decoder_cell = tf.contrib.rnn.MultiRNNCell([decoder_cell] * self._layers,
                                                             state_is_tuple=True)
        # Target Vocab embedding
        with tf.name_scope('decoder_embedding'):
            # vocab_size - 1, manually add zero tensor for PADDING embeddings
            self._decoder_embeddings = tf.get_variable(
                name="decoder_embeddings",
                initializer=tf.truncated_normal([self._vocab_manager.vocab_target_len - 1, self._embedding_dim],
                                                stddev=0.5)
            )

            # zero embedding for <PAD>
            pad_embeddings = tf.get_variable(initializer=tf.zeros([1, self._embedding_dim]),
                                             name='decoder_pad_embedding',
                                             trainable=False)
            self._decoder_embeddings = tf.concat(values=[pad_embeddings, self._decoder_embeddings], axis=0)

            # replace vocab-id with embedding
            # self._embedded shape is self._batch_size * self._max_length_encode * self._embedding_dim
            self._decoder_embedded = tf.nn.embedding_lookup(self._decoder_embeddings, self._decoder_inputs)

        with tf.variable_scope('decoder'):
            self._decoder_outputs, self._decoder_states = tf.nn.dynamic_rnn(self._decoder_cell,
                                                                            initial_state=self._encoder_states,
                                                                            inputs=self._decoder_embedded,
                                                                            dtype=tf.float32)

    def _calculate_attention(self, decoder_hidden_states, num):
        """
        Set up attention layer
        :return: attentioned_outputs
        """
        attentioned_decoder_outputs = list()

        def _attention(batch_hidden_states):
            """
            Calculate batch attention hidden states
            :param batch_hidden_states: shape: [batch_size, hidden_dim]
            :return:
            """
            # dot product
            # [batch_size, hidden_dim] => [batch_size*hidden_dim, 1]
            reshaped_batch_hidden_states = tf.reshape(batch_hidden_states, shape=[
                tf.multiply(
                    tf.shape(batch_hidden_states)[0],
                    tf.shape(batch_hidden_states)[1]
                ),
                1
            ])

            # [batch_size, max_length_encode, hidden_dim] => [batch_size*hidden_dim, max_length_encode]
            reshaped_encoder_outputs = tf.reshape(tf.transpose(self._encoder_outputs, perm=[0, 2, 1]), shape=[
                tf.multiply(
                    tf.shape(self._encoder_outputs)[0],
                    tf.shape(self._encoder_outputs)[2]
                ),
                tf.shape(self._encoder_outputs)[1]
            ])

            # [batch_size*hidden_dim, max_length_encode]
            element_wise_multiply = tf.multiply(reshaped_batch_hidden_states, reshaped_encoder_outputs)

            # [batch_size, max_length_encode, hidden_dim]
            recover_shape = tf.transpose(tf.reshape(element_wise_multiply,
                                                    shape=[
                                                        tf.shape(self._encoder_outputs)[0],
                                                        tf.shape(self._encoder_outputs)[2],
                                                        tf.shape(self._encoder_outputs)[1]]),
                                         perm=[0, 2, 1])

            # [batch_size, max_length_encode]
            dot_product = tf.reduce_sum(recover_shape, axis=2)

            # softmax weight
            softmax_weight = tf.nn.softmax(dot_product)

            # weighted sum [batch_size, max_length_encoder] => [batch_size, max_length_encoder, 1]
            expanded_softmax_weight = tf.expand_dims(softmax_weight, 2)

            # context vector for hidden_state
            weight_encoder_hidden_state = tf.multiply(expanded_softmax_weight, self._encoder_outputs)
            # [batch_size, hidden_dim]
            context_vector = tf.reduce_sum(weight_encoder_hidden_state, axis=1)

            # W1*ht
            weighted_decoder_hidden_state = tf.map_fn(
                lambda x: tf.reduce_sum(tf.matmul(self._attention_weight1, tf.expand_dims(x, 1)), axis=1),
                elems=batch_hidden_states
            )
            # W2*ct
            weighted_context_vector = tf.map_fn(
                lambda x: tf.reduce_sum(tf.matmul(self._attention_weight2, tf.expand_dims(x, 1)), axis=1),
                elems=context_vector
            )

            attention_hidden_state = tf.tanh(tf.add(weighted_decoder_hidden_state, weighted_context_vector))
            return attention_hidden_state

        for batch_hidden_state in tf.unstack(decoder_hidden_states, num, axis=1):
            attentioned_decoder_outputs.append(_attention(batch_hidden_state))

        return tf.stack(attentioned_decoder_outputs, axis=1)

    def _build_train_feed(self, batch):
        feed_dict = dict()
        feed_dict[self._encoder_inputs] = batch.encoder_seq
        feed_dict[self._encoder_input_keep_prob] = 1. - self._dropout
        feed_dict[self._encoder_output_keep_prob] = 1. - self._dropout
        feed_dict[self._decoder_input_keep_prob] = 1. - self._dropout
        feed_dict[self._decoder_output_keep_prob] = 1. - self._dropout
        feed_dict[self._decoder_inputs] = batch.decoder_seq
        feed_dict[self._decoder_targets] = batch.target_seq
        return feed_dict

    def _build_test_feed(self, inputs):
        feed_dict = dict()
        feed_dict[self._encoder_output_keep_prob] = 1.
        feed_dict[self._encoder_input_keep_prob] = 1.
        feed_dict[self._decoder_input_keep_prob] = 1.
        feed_dict[self._decoder_output_keep_prob] = 1.
        feed_dict[self._encoder_inputs] = inputs
        return feed_dict

    def loss(self, batch):
        assert self._is_test == False

        feed_dict = self._build_train_feed(batch)

        return self._loss, feed_dict

    def optimize(self, batch):
        assert self._is_test == False

        feed_dict = self._build_train_feed(batch)
        return self._optimizer, feed_dict

    def train(self, batch):

        assert self._is_test == False

        feed_dict = self._build_train_feed(batch)
        return self._predictions, self._loss, self._optimizer, feed_dict

    def predict(self, inputs):
        """
        When testing, LSTM decoder had to predict token step by step
        :param self:
        :param inputs:
        :return:
        """
        assert self._is_test == True

        feed_dict = self._build_test_feed(inputs)
        return self._predictions, self._decoder_states, feed_dict
