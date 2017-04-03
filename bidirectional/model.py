# coding=utf8

import util
from data_provider.data_iterator import VocabManager
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import numpy as np


class Model:
    BEAM_MAX_SCORE = 1e10
    BEAM_MIN_SCORE = -1e10

    epsilon = 1e-5

    def __init__(self, vocab_manager, opts, is_test):
        self._hidden_dim = util.get_value(opts, "hidden_dim", 128)
        self._dropout = util.get_value(opts, "dropout", 0.25)
        self._layers = util.get_value(opts, "layer", 1)
        self._max_length_encode = util.get_value(opts, "max_length_encode", 20)
        self._max_length_decode = util.get_value(opts, "max_length_decode", 20)
        self._vocab_manager = vocab_manager
        self._embedding_dim = util.get_value(opts, "embedding_dim", 128)
        self._learning_rate = util.get_value(opts, "learning_rate", 0.01)
        self._gradient_clip = util.get_value(opts, "gradient_clip", 5)

        self._batch_size = util.get_value(opts, "batch_size", 20)
        self._beam_size = util.get_value(opts, "beam_size", 5)

        self._uniform_init_min = util.get_value(opts, "params_init_min", -0.08)
        self._uniform_init_max = util.get_value(opts, "params_init_max", 0.08)

        self._is_test = is_test

        self._test_batch_size = util.get_value(opts, "test_batch_size", 1)
        self._is_cell_share = util.get_value(opts, "is_cell_share")
        self._optimizer_choice = util.get_value(opts, "optimizer", "rmsprop")

        if self._is_test:
            self._build_test_graph()
        else:
            self._build_train_graph()

    def _build_train_graph(self):
        self._build_encoder()
        self._build_train_decoder()

        with tf.variable_scope("attention"):
            self._attention_weight1 = tf.get_variable(
                initializer=tf.random_uniform(shape=[self._hidden_dim * 2, self._hidden_dim * 2],
                                              minval=self._uniform_init_min,
                                              maxval=self._uniform_init_max),
                name="weight_1"
            )
            self._attention_weight2 = tf.get_variable(
                initializer=tf.random_uniform(shape=[self._hidden_dim * 2, self._hidden_dim * 2],
                                              minval=self._uniform_init_min,
                                              maxval=self._uniform_init_max),
                name="weight_2"
            )

        with tf.variable_scope("softmax") as scope:
            self._softmax_weight = tf.get_variable(
                initializer=tf.truncated_normal([self._vocab_manager.vocab_target_len, self._hidden_dim * 2],
                                                stddev=0.5),
                name="softmax_weight"
            )
            self._softmax_bias = tf.get_variable(
                initializer=tf.truncated_normal([self._vocab_manager.vocab_target_len], stddev=0.5),
                name="softmax_bias"
            )

        self._attention_decoder_outputs = self._calculate_attention(self._decoder_outputs, num=self._max_length_decode)

        softmax_outputs = self._normalize(self._attention_decoder_outputs)

        self._predictions = self._predict(softmax_outputs)

        softmax_outputs = tf.add(softmax_outputs,
                                 tf.constant(self.epsilon, dtype=tf.float32))

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
                tf.log(self._labeled), axis=1
            )

            self._loss = tf.negative(tf.reduce_mean(self._predict_log_probability))

            # tf.summary.scalar('loss', self._loss)

        with tf.name_scope('back_propagation'):

            if self._optimizer_choice == "rmsprop":
                print("Rmsprop Optimizer")
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self._learning_rate, decay=0.95)
            else:
                print("Adam Optimizer")
                optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
            # self._optimizer = optimizer.minimize(self._loss)

            # clipped at 5 to alleviate the exploding gradient problem
            self._gvs = optimizer.compute_gradients(self._loss)
            self._capped_gvs = [(tf.clip_by_value(grad, -self._gradient_clip, self._gradient_clip), var) for grad, var in self._gvs]
            self._optimizer = optimizer.apply_gradients(self._capped_gvs)

    def _build_test_graph(self):
        """
        build test graph
        :return:
        """
        self._build_encoder()

        with tf.variable_scope("attention"):
            self._attention_weight1 = tf.get_variable(
                initializer=tf.random_uniform(shape=[self._hidden_dim * 2, self._hidden_dim * 2],
                                              minval=self._uniform_init_min,
                                              maxval=self._uniform_init_max),
                name="weight_1"
            )
            self._attention_weight2 = tf.get_variable(
                initializer=tf.random_uniform(shape=[self._hidden_dim * 2, self._hidden_dim * 2],
                                              minval=self._uniform_init_min,
                                              maxval=self._uniform_init_max),
                name="weight_2"
            )

        with tf.variable_scope("softmax") as scope:
            self._softmax_weight = tf.get_variable(
                initializer=tf.truncated_normal([self._vocab_manager.vocab_target_len, self._hidden_dim * 2],
                                                stddev=0.5),
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
            decoder_cell = LSTMCell(num_units=self._hidden_dim * 2, state_is_tuple=True)
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

        # decoder
        with tf.variable_scope('decoder'):

            """
            Batch_size = 1
            """
            batch_size = self._test_batch_size

            # Setup
            decoder_embedded = tf.nn.embedding_lookup(self._decoder_embeddings, [[VocabManager.GO_TOKEN_ID]])
            decoder_outputs, self._decoder_states = tf.nn.dynamic_rnn(self._decoder_cell,
                                                                      initial_state=self._encoder_states,
                                                                      inputs=decoder_embedded,
                                                                      dtype=tf.float32)
            attention_decoder_outputs = self._calculate_attention(decoder_outputs, num=1)
            softmax_outputs = self._normalize(attention_decoder_outputs)
            symbols_probs, index = tf.nn.top_k(softmax_outputs, k=self._beam_size)

            self._predictions = tf.multiply(
                tf.constant(np.array([[1] + [0] * (self._max_length_decode - 1)] * self._beam_size),
                            dtype=tf.int64),
                tf.cast(tf.reshape(index, [self._beam_size, 1]), dtype=tf.int64)
            )

            self._first_predictions = tf.cast(tf.reshape(index, [self._beam_size, 1]), dtype=tf.int64)

            logprobs = tf.reshape(tf.log(symbols_probs), [self._beam_size])

            # Reconstruct LSTMStateTuple for encoder_states

            def _reconstruct_LSTMStateTuple(hidden_states):
                """
                Expand LSTMStateTuple
                """
                new_tuples = []
                template = tf.ones([self._beam_size, self._hidden_dim * 2], dtype=tf.float32)
                for ls in tf.unstack(hidden_states, axis=0):
                    extended = list()
                    for t in tf.unstack(ls, axis=0):
                        extended.append(tf.multiply(template, t))
                    new_tuples.append(
                        LSTMStateTuple(extended[0], extended[1])
                    )
                return tuple(new_tuples)

            def _reorder_LSTMStateTuple(hidden_states, indices):
                """
                Reorder LSTMState
                """
                new_tuples = []
                for ls in tf.unstack(hidden_states, axis=0):
                    extended = list()
                    for t in tf.unstack(ls, axis=0):
                        extended.append(tf.gather(t, indices))
                    new_tuples.append(
                        LSTMStateTuple(extended[0], extended[1])
                    )
                return tuple(new_tuples)

            self._decoder_states = _reconstruct_LSTMStateTuple(self._decoder_states)

            # Reconstruct Encoder Output
            new_encoder_outputs = list()
            for h in tf.unstack(self._encoder_outputs, axis=1):
                template = tf.ones([self._beam_size, self._hidden_dim * 2], dtype=tf.float32)
                new_encoder_outputs.append(tf.multiply(template, h))
            self._encoder_outputs = tf.stack(new_encoder_outputs, axis=1)

            def _loop_body(token_id, curr_ts, _predictions, _logprobs, _mask, states):
                """
                :param token_id:  last_predictions
                :param curr_ts:   time_step
                :param _predictions: sequences
                :param _logprobs: log probabilities of each sequence in beam,
                :param _mask:     Mask to indicate whether the sequence finishes or not
                :param states:    LSTM hidden states
                :return:
                """
                _decoder_embedded = tf.nn.embedding_lookup(self._decoder_embeddings, token_id)

                _decoder_outputs, _decoder_states = tf.nn.dynamic_rnn(self._decoder_cell,
                                                                      initial_state=states,
                                                                      inputs=_decoder_embedded,
                                                                      dtype=tf.float32)
                _attention_decoder_outputs = self._calculate_attention(_decoder_outputs, num=1)

                _softmax_outputs = self._normalize(_attention_decoder_outputs)

                # 1. Predict next tokens

                _symbols, symbols_logprobs, parent_ref = self._beam_predict(_softmax_outputs, _logprobs, _mask)

                # 2. Reorder mask

                # [beam_size]
                _mask = tf.gather(_mask, parent_ref)

                # 3. Update log probability

                # [beam_size]
                _logprobs = tf.gather(_logprobs, parent_ref)

                # Calculate log probability of each sequence in beam, then reshape to [batch_size*beam_size]
                _logprobs = tf.add(
                    _logprobs,
                    tf.multiply(
                        tf.cast(
                            tf.subtract(tf.constant(1, dtype=tf.int64), _mask),
                            dtype=tf.float32
                        ),
                        symbols_logprobs
                    )
                )

                # 4. Update individual sequences in beam

                # [beam_size, max_sequence_length]
                predictions = tf.gather(_predictions, parent_ref)

                ts = tf.cast(tf.equal(tf.range(self._max_length_decode), curr_ts), dtype=tf.int64)
                reshaped_ts = tf.tile(tf.expand_dims(ts, 0), [self._beam_size, 1])
                # [beam_size, max_sequence_length]
                ts_value = tf.multiply(reshaped_ts,
                                       tf.cast(tf.reshape(_symbols, [-1, 1]), dtype=tf.int64))
                predictions = tf.add(predictions, ts_value)

                # 5. Update mask
                _mask = tf.cast(
                    tf.equal(
                        tf.cast(
                            _symbols,
                            dtype=tf.int64
                        ),
                        tf.constant(VocabManager.EOS_TOKEN_ID, dtype=tf.int64)
                    ),
                    dtype=tf.int64
                )

                # 6. Reorder LSTMStates
                _decoder_states = _reorder_LSTMStateTuple(_decoder_states, parent_ref)

                # 6. Update curr_ts
                curr_ts = tf.add(curr_ts, 1)

                token_ids = tf.cast(tf.reshape(_symbols, [self._beam_size, 1]), dtype=tf.int64)
                return token_ids, curr_ts, predictions, _logprobs, _mask, _decoder_states

            def _terminate_condition(token_id, curr_ts, predictions, logprobs, _mask, states):
                """
                :return:
                """
                return tf.logical_and(
                    tf.less(curr_ts, self._max_length_decode),
                    tf.less(tf.reduce_sum(_mask), self._beam_size)
                )

            self._last_prediction, time_steps, self._predictions, self._logprobs, self._mask, self._decoder_states = tf.while_loop(
                _terminate_condition,
                _loop_body,
                [
                    self._first_predictions,
                    tf.constant(1),
                    # prediction, shape: [beam_size, length],
                    self._predictions,
                    # log probability of each sequence in beam, [beam_size],
                    logprobs,
                    # mask, indicate which sequence in beam finish
                    tf.zeros([self._beam_size], dtype=tf.int64),
                    self._decoder_states
                ],
                parallel_iterations=1
            )

    def _predict(self, outputs):
        """
        prediction
        :param outputs:
        :return:
        """
        with tf.name_scope("prediction"):
            predictions = tf.arg_max(outputs, dimension=2)
        return predictions

    def _beam_predict(self, outputs, prev_logprobs, mask):
        """
        Only support batch size = 1
        Only used in test
        Beam predict, Calculate Beam Score
        :param outputs: shape: [beam_size, 1, self._vocab_manager.target_len]
                        type: tensor
        :param prev_logprobs:  [batch_size*beam_size],
                               type: tensor
        :param mask: [batch_size*beam_size], indicate which sequence in beam finished,
                     type: tensor
        :return:
            Symbols idx, shape: [beam_size]
            Symbols log probabilities, shape: [beam_size]
            Parent_ref, (symbols idx belongs to which sequence in beam), shape: [beam_size]
        """
        reshaped_outputs = tf.reshape(outputs, shape=[self._beam_size, self._vocab_manager.vocab_target_len])
        log_reshaped_outputs = tf.log(reshaped_outputs)
        logprobs_tensor = tf.reshape(prev_logprobs, [-1, 1])

        # shape: [beam_size, vocab_target_len]
        extended_logprobs = tf.add(
            tf.zeros([self._beam_size, self._vocab_manager.vocab_target_len]),
            logprobs_tensor
        )
        curr_logprobs = tf.add(log_reshaped_outputs, extended_logprobs)

        mask_tensor = tf.cast(tf.reshape(mask, shape=[-1, 1]), dtype=tf.float32)
        done_one_hot_tensor = tf.multiply(
            tf.cast(
                tf.equal(tf.range(self._vocab_manager.vocab_target_len), VocabManager.EOS_TOKEN_ID), tf.float32
            ),
            tf.constant(self.BEAM_MAX_SCORE, dtype=tf.float32)
        )
        done_vector = tf.multiply(
            tf.cast(
                tf.not_equal(tf.range(self._vocab_manager.vocab_target_len), VocabManager.EOS_TOKEN_ID),
                tf.float32
            ),
            tf.constant(self.BEAM_MIN_SCORE, dtype=tf.float32)
        )
        done_vector = tf.add(done_one_hot_tensor, done_vector)
        expanded_one_hot = tf.tile(tf.expand_dims(done_vector, 0), [self._beam_size, 1])

        expanded_mask = tf.reshape(
            tf.multiply(expanded_one_hot, mask_tensor),
            [
                self._beam_size,
                self._vocab_manager.vocab_target_len
            ]
        )
        curr_logprobs = tf.add(curr_logprobs, expanded_mask)

        beam_logprobs, indices = tf.nn.top_k(
            tf.reshape(curr_logprobs, [1, self._beam_size * self._vocab_manager.vocab_target_len]),
            self._beam_size
        )

        symbols = tf.reshape(
            tf.mod(indices, self._vocab_manager.vocab_target_len),
            [self._beam_size]
        )
        parent_refs = tf.reshape(
            tf.div(indices, self._vocab_manager.vocab_target_len),
            [self._beam_size]
        )

        symbols_logprobs = tf.gather(
            tf.reshape(log_reshaped_outputs, [self._beam_size * self._vocab_manager.vocab_target_len]),
            tf.reshape(indices, [self._beam_size])
        )

        return symbols, symbols_logprobs, parent_refs

    def _normalize(self, outputs):
        """
        Normalize, Weighted Softmax
        :param outputs:
        :return:
        """
        # softmax output
        # weighted_output shape: batch_size * max_length_decoder * vocab_f_len
        self._weighted_outputs = tf.map_fn(
            lambda x: tf.add(tf.matmul(x, self._softmax_weight, transpose_b=True), self._softmax_bias),
            outputs)
        softmax_outpus = tf.nn.softmax(self._weighted_outputs)
        return softmax_outpus

    def _build_encoder(self):
        """
        Build Encoder
        :return:
        """
        # Encoder placeholder
        with tf.name_scope('encoder_placeholder'):
            # encoder inputs are integers, representing the index of words in the sentences
            self._encoder_fw_inputs = tf.placeholder(tf.int32, [None, self._max_length_encode], name="fw_inputs")
            self._source_length = tf.placeholder(tf.int32, [None], name="source_length")
            self._encoder_output_keep_prob = tf.placeholder(tf.float32, name="output_keep_prob")
            self._encoder_input_keep_prob = tf.placeholder(tf.float32, name="input_keep_prob")

        # source vocab embedding
        with tf.name_scope('encoder_embedding'):
            # vocab_size - 1, manually add zero tensor for PADDING embeddings
            self._encoder_embeddings = tf.get_variable(
                initializer=tf.truncated_normal([self._vocab_manager.vocab_source_len - 1, self._embedding_dim],
                                                stddev=0.5),
                name="encoder_embeddings"
            )

            # zero embedding for <PAD>
            pad_embeddings = tf.get_variable(initializer=tf.zeros([1, self._embedding_dim]),
                                             name='encoder_pad_embedding', trainable=False)
            self._encoder_embeddings = tf.concat(values=[pad_embeddings, self._encoder_embeddings], axis=0)

            # replace vocab-id with embedding
            self._encoder_fw_embedded = tf.nn.embedding_lookup(self._encoder_embeddings, self._encoder_fw_inputs)

        with tf.variable_scope('encoder'):
            # dropout is only available during training
            # define encoder cell
            if not self._is_cell_share:
                with tf.variable_scope("encoder_fw_cell"):
                    encoder_fw_cell = LSTMCell(
                        num_units=self._hidden_dim,
                        state_is_tuple=True
                    )
                    encoder_fw_cell = tf.contrib.rnn.DropoutWrapper(encoder_fw_cell,
                                                                    input_keep_prob=self._encoder_input_keep_prob,
                                                                    output_keep_prob=self._encoder_output_keep_prob)
                    encoder_fw_cell = tf.contrib.rnn.MultiRNNCell([encoder_fw_cell] * self._layers,
                                                                  state_is_tuple=True)

                with tf.variable_scope("encoder_bw_cell"):
                    encoder_bw_cell = LSTMCell(num_units=self._hidden_dim, state_is_tuple=True)
                    encoder_bw_cell = tf.contrib.rnn.DropoutWrapper(encoder_bw_cell,
                                                                    input_keep_prob=self._encoder_input_keep_prob,
                                                                    output_keep_prob=self._encoder_output_keep_prob)
                    encoder_bw_cell = tf.contrib.rnn.MultiRNNCell([encoder_bw_cell] * self._layers,
                                                                  state_is_tuple=True)
            else:
                with tf.variable_scope("encoder_cell"):
                    encoder_cell = LSTMCell(
                        num_units=self._hidden_dim,
                        state_is_tuple=True
                    )
                    encoder_cell = tf.contrib.rnn.DropoutWrapper(encoder_cell,
                                                                 input_keep_prob=self._encoder_input_keep_prob,
                                                                 output_keep_prob=self._encoder_output_keep_prob)
                    encoder_fw_cell = tf.contrib.rnn.MultiRNNCell([encoder_cell] * self._layers,
                                                                  state_is_tuple=True)
                    encoder_bw_cell = tf.contrib.rnn.MultiRNNCell([encoder_cell] * self._layers,
                                                                  state_is_tuple=True)

            with tf.variable_scope("bidirectional_encode"):

                (self._fw_encoder_outputs, self._bw_encoder_outputs), (self._fw_encoder_states, self._bw_encoder_states) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=encoder_fw_cell,
                    cell_bw=encoder_bw_cell,
                    inputs=self._encoder_fw_embedded,
                    sequence_length=self._source_length,
                    dtype=tf.float32
                )

            # concat outputs, [batch_size, max_length_encode, hidden_state * 2]
            self._encoder_outputs = tf.concat((self._fw_encoder_outputs, self._bw_encoder_outputs), axis=2)

            def _reconstruct_LSTMStateTuple(fw_states, bw_states):
                """
                Concatenate the fw_state and bw_state
                """
                new_tuples = []
                for (fw, bw) in zip(fw_states, bw_states):
                    new_tuples.append(
                        LSTMStateTuple(
                            c=tf.concat((fw.c, bw.c), axis=1),
                            h=tf.concat((fw.h, bw.h), axis=1)
                        )
                    )
                return tuple(new_tuples)

            self._encoder_states = _reconstruct_LSTMStateTuple(self._fw_encoder_states, self._bw_encoder_states)

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
            decoder_cell = LSTMCell(num_units=(self._hidden_dim * 2), state_is_tuple=True)
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
            :param batch_hidden_states: shape: [batch_size, hidden_dim*2]
            :return:
            """
            # dot product
            # [batch_size, hidden_dim*2] => [batch_size*hidden_dim*2, 1]
            reshaped_batch_hidden_states = tf.reshape(batch_hidden_states, shape=[
                tf.multiply(
                    tf.shape(batch_hidden_states)[0],
                    tf.shape(batch_hidden_states)[1]
                ),
                1
            ])

            # [batch_size, max_length_encode, hidden_dim*2] => [batch_size*hidden_dim*2, max_length_encode]
            reshaped_encoder_outputs = tf.reshape(tf.transpose(self._encoder_outputs, perm=[0, 2, 1]), shape=[
                tf.multiply(
                    tf.shape(self._encoder_outputs)[0],
                    tf.shape(self._encoder_outputs)[2]
                ),
                tf.shape(self._encoder_outputs)[1]
            ])

            # [batch_size*hidden_dim*2, max_length_encode]
            element_wise_multiply = tf.multiply(reshaped_batch_hidden_states, reshaped_encoder_outputs)

            # [batch_size, max_length_encode, hidden_dim*2]
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
            # [batch_size, hidden_dim*2]
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
        feed_dict[self._encoder_fw_inputs] = batch.encoder_seq
        feed_dict[self._encoder_input_keep_prob] = 1. - self._dropout
        feed_dict[self._encoder_output_keep_prob] = 1. - self._dropout
        feed_dict[self._decoder_input_keep_prob] = 1. - self._dropout
        feed_dict[self._decoder_output_keep_prob] = 1. - self._dropout
        feed_dict[self._decoder_inputs] = batch.decoder_seq
        feed_dict[self._decoder_targets] = batch.target_seq
        feed_dict[self._source_length] = batch.source_length
        return feed_dict

    def _build_test_feed(self, inputs, source_length):
        feed_dict = dict()
        feed_dict[self._encoder_output_keep_prob] = 1.
        feed_dict[self._encoder_input_keep_prob] = 1.
        feed_dict[self._decoder_input_keep_prob] = 1.
        feed_dict[self._decoder_output_keep_prob] = 1.
        feed_dict[self._encoder_fw_inputs] = inputs
        feed_dict[self._source_length] = source_length
        return feed_dict

    def loss(self, batch):
        assert self._is_test == False

        feed_dict = self._build_train_feed(batch)

        return self._predictions, self._loss, feed_dict

    def optimize(self, batch):
        assert self._is_test == False

        feed_dict = self._build_train_feed(batch)
        return self._optimizer, feed_dict

    def train(self, batch):

        assert self._is_test == False

        feed_dict = self._build_train_feed(batch)
        return self._predictions, self._loss, self._gvs, self._capped_gvs, self._optimizer, feed_dict

    def predict(self, inputs, source_length):
        """
        When testing, LSTM decoder had to predict token step by step
        :param source_length:
        :param reverse_encoder_seq:
        :param self:
        :param inputs:
        :return:
        """
        assert self._is_test == True

        feed_dict = self._build_test_feed(inputs, source_length)
        return self._last_prediction, self._predictions, self._logprobs, self._mask, self._decoder_states, feed_dict

    def encode(self, batch):
        if self._is_test:
            feed_dict = self._build_test_feed(batch.encoder_seq, batch.source_length)
        else:
            feed_dict = self._build_train_feed(batch)
        return self._encoder_outputs, self._encoder_states, feed_dict
