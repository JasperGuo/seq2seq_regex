# coding=utf8

import util
from data_provider.data_iterator import VocabManager
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import numpy as np


def update_output(index, value, outputs):
    """
    Insert value to outputs, with respect to index
    :param index: scalar
    :param value:   [shape_0, shape_1]
    :param outputs,   [shape_0, length, shape_1]
    :return:          [shape_0, length, shape_1]
    """
    shape_0 = tf.shape(outputs)[0]
    shape_1 = tf.shape(outputs)[2]
    max_length = tf.shape(outputs)[1]

    ts_one_hot_vector = tf.cast(tf.equal(tf.range(max_length), index), dtype=tf.float32)
    reshaped_ts = tf.reshape(
        tf.tile(
            tf.expand_dims(ts_one_hot_vector, 0),
            [shape_0, 1]
        ),
        shape=[shape_0, max_length, 1]
    )
    # [shape_0, length, shape_1]
    ts_value = tf.multiply(
        reshaped_ts,
        tf.reshape(
            value,
            shape=[shape_0, 1, shape_1]
        )
    )
    return tf.add(outputs, ts_value)


def update_tape(index, value, tapes):
    """
    Insert value to tapes, with respect to index
    :param index: scalar
    :param value:   [shape_0, shape_1, shape_2]
    :param tapes,   [shape_0, length, shape_1, shape_2]
    :return:        [shape_0, length, shape_1, shape_2]
    """
    shape_0 = tf.shape(tapes)[0]
    shape_1 = tf.shape(tapes)[2]
    shape_2 = tf.shape(tapes)[3]
    max_length = tf.shape(tapes)[1]

    ts_one_hot_vector = tf.cast(tf.equal(tf.range(max_length), index), dtype=tf.float32)
    reshaped_ts = tf.reshape(
        tf.tile(
            tf.expand_dims(ts_one_hot_vector, 0),
            [shape_0, 1]
        ),
        shape=[shape_0, max_length, 1, 1]
    )
    # [shape_0, length, shape_1, shape_2]
    ts_value = tf.multiply(
        reshaped_ts,
        tf.reshape(
            value,
            shape=[shape_0, 1, shape_1, shape_2]
        )
    )
    return tf.add(tapes, ts_value)


class LSTMN:
    """
    Refer to https://arxiv.org/pdf/1601.06733.pdf
    """

    def __init__(self,
                 cells,
                 batch_size,
                 hidden_size,
                 embedding_size,
                 max_length,
                 scope_name=None,
                 scope=None,
                 initial_states=None):
        """
        Init LSTMCells
        :param cells
        :param batch_size:
        :param hidden_size:
        :param embedding_size
        :param max_length
        :param scope: variable scope
        :param initial_states: (LSTMStateTuple, )
        """
        self._batch_size = batch_size
        self._hidden_size = hidden_size
        self._embedding_size = embedding_size
        self._scope = scope
        self._scope_name = scope_name
        self._cells = cells
        self._num_layer = len(cells)
        self._max_length = max_length

        self._last_attentive_hidden_state = tf.zeros([self._batch_size, self._hidden_size], dtype=tf.float32)

        self._ts = tf.constant(0, dtype=tf.int32)

        self._init_intra_attention_params()

        self._is_set_reuse = False

        if initial_states:
            self._init_states = initial_states
            self._init_states_length = len(self._init_states)
        else:
            template_tuple = LSTMStateTuple(
                h=tf.zeros([self._batch_size, self._hidden_size], dtype=tf.float32),
                c=tf.zeros([self._batch_size, self._hidden_size], dtype=tf.float32)
            )
            self._init_states = [template_tuple]
            self._init_states_length = 0

    def _init_intra_attention_params(self):
        """
        Initialize intra-attention parameters
        :return:
        """
        with tf.variable_scope('intra_attention_weight'):
            # W_h
            self._intra_attention_weight1 = tf.get_variable(
                initializer=tf.truncated_normal([self._hidden_size, self._hidden_size], stddev=0.5),
                name="intra_attention_weight1"
            )

            # W_hat(h)
            self._intra_attention_weight2 = tf.get_variable(
                initializer=tf.truncated_normal([self._hidden_size, self._hidden_size], stddev=0.5),
                name="intra_attention_weight2"
            )

            # W_X
            self._intra_attention_weight_word = tf.get_variable(
                initializer=tf.truncated_normal([self._hidden_size, self._embedding_size], stddev=0.5),
                name="intra_attention_weight_word"
            )

            if self._num_layer > 1:
                # W_l, used when there are multiple layers
                self._intra_attention_weight_l = tf.get_variable(
                    initializer=tf.truncated_normal([self._hidden_size, self._hidden_size], stddev=0.5),
                    name="intra_attention_weight_l"
                )

            # v
            self._intra_attention_weight_vector = tf.get_variable(
                initializer=tf.truncated_normal([self._hidden_size], stddev=0.5),
                name="intra_attention_weight_vector"
            )

    def _calc_intra_attention(self, ts, inputs, hidden_states, memory_tapes, last_attentive_hs, input_weighted_matrix):
        """
        Calculate intra attention
        The size of inputs and input_weighted_matrix has to match!!
        :param ts                           scalar
        :param inputs:                      [batch_size, 1, embedding_size] | [batch_size, 1, hidden_size]
        :param hidden_states:               [batch_size, max_length, hidden_dim]
        :param memory_tapes:                [batch_size, max_length, hidden_dim]
        :param last_attentive_hs:           [batch_size, hidden_size]
        :param input_weighted_matrix:       [hidden_size, embedding_size] | [hidden_size, hidden_size]
        :return:
        """
        with tf.name_scope("calc_intra_attention"):
            previous_hidden_states = tf.slice(hidden_states, [0, 0, 0], [self._batch_size, ts, self._hidden_size])

            # W_h * h_i
            weighted_previous_hidden_states = tf.reshape(
                tf.map_fn(
                    lambda x: tf.matmul(self._intra_attention_weight1, x, transpose_b=True),
                    tf.reshape(
                        previous_hidden_states,
                        [tf.multiply(self._batch_size, ts), 1, self._hidden_size]
                    )
                ),
                [self._batch_size, ts, self._hidden_size]
            )

            # W_x * x_t
            weighted_inputs = tf.reshape(
                tf.map_fn(lambda x: tf.matmul(input_weighted_matrix, x, transpose_b=True), inputs),
                shape=[self._batch_size, self._hidden_size]
            )

            # W_hat(h) * hat(h)_t-1
            weighted_last_attentive_hidden_state = tf.reshape(
                tf.map_fn(
                    lambda x: tf.matmul(self._intra_attention_weight2, x, transpose_b=True),
                    tf.reshape(
                        last_attentive_hs,
                        [self._batch_size, 1, self._hidden_size]
                    )
                ),
                shape=[self._batch_size, self._hidden_size]
            )

            # add
            sum_of_word_hidden_states = tf.add(weighted_inputs, weighted_last_attentive_hidden_state)

            weighted_sum = tf.transpose(
                tf.map_fn(
                    lambda x: tf.add(x, sum_of_word_hidden_states),
                    tf.transpose(
                        weighted_previous_hidden_states,
                        perm=[1, 0, 2]
                    )
                ),
                perm=[1, 0, 2]
            )

            reshaped_v = tf.reshape(self._intra_attention_weight_vector, [1, self._hidden_size])
            # a
            a = tf.reshape(
                tf.map_fn(
                    lambda x: tf.matmul(reshaped_v, x, transpose_b=True),
                    tf.reshape(tf.tanh(weighted_sum), [tf.multiply(self._batch_size, ts), 1, self._hidden_size])
                ),
                shape=[self._batch_size, ts]
            )

            # s, softmax
            softmax_weight = tf.nn.softmax(a, dim=-1)

            # attentive hidden state, [batch_size, hidden_size]
            attentive_hidden_state = tf.reshape(
                tf.reduce_sum(
                    tf.multiply(
                        previous_hidden_states,
                        tf.reshape(softmax_weight, [self._batch_size, ts, 1])
                    ),
                    axis=1
                ),
                shape=[self._batch_size, self._hidden_size]
            )

            # attentive memory cell, [batch_size, hidden_size]
            previous_memory_cells = tf.slice(memory_tapes, [0, 0, 0], [self._batch_size, ts, self._hidden_size])
            attentive_memory_cell = tf.reshape(
                tf.reduce_sum(
                    tf.multiply(
                        previous_memory_cells,
                        tf.reshape(softmax_weight, [self._batch_size, ts, 1])
                    ),
                    axis=1
                ),
                shape=[self._batch_size, self._hidden_size]
            )

            return attentive_hidden_state, attentive_memory_cell

    def step(self, ts, inputs, hidden_states, memory_tapes, last_attentive_hs):
        """
        Run one step
        :param last_attentive_hs: [batch_size, hidden_size]
        :param ts               scalar
        :param memory_tapes:    [batch_size, max_length, num_layer, hidden_size]
        :param hidden_states:   [batch_size, max_length, num_layer, hidden_size]
        :param inputs: [batch_size, 1, embedding_size]
        :return:
            outputs: [batch_size, hidden_dim],
            hidden_states: [batch_size, num_layer, hidden_size]
            memory_tapes:  [batch_size, num_layer, hidden_size],
            last_attentive_hs: [batch_size, hidden_dim]
        """

        def _get_tape(layer, tape):
            """
            Get state according to layer
            :param layer: scalar
            :param tape:  [batch_size, max_length, num_layer, hidden_dim]
            :return:      [batch_size, max_length, hidden_dim]
            """
            return tf.reshape(
                tf.slice(tape, [0, 0, layer, 0], [self._batch_size, self._max_length, 1, self._hidden_size]),
                shape=[self._batch_size, self._max_length, self._hidden_size]
            )

        def _run_with_init_states():
            """
            Run with init_states, no intra-attention is calculated
            :return:
            """

            outputs, states = tf.nn.dynamic_rnn(
                cell=self._cells[0],
                initial_state=self._init_states[0],
                inputs=inputs,
                dtype=tf.float32
            )

            new_hidden_states = tf.zeros([self._batch_size, self._num_layer, self._hidden_size], dtype=tf.float32)
            new_memory_tapes = tf.zeros([self._batch_size, self._num_layer, self._hidden_size], dtype=tf.float32)
            new_hidden_states = update_output(0, states.h, new_hidden_states)
            new_memory_tapes = update_output(0, states.c, new_memory_tapes)

            _inputs = states.h

            if not self._is_set_reuse:
                self._scope.reuse_variables()
                self._is_set_reuse = True

            for i, cell in enumerate(self._cells[1:]):
                curr_layer = tf.constant((i + 1))
                _layer_hidden_state, _layer_lstm_state_tuple = tf.nn.dynamic_rnn(
                    cell=cell,
                    initial_state=self._init_states[i + 1],
                    inputs=_inputs,
                    dtype=tf.float32
                )
                new_hidden_states = update_output(curr_layer, _layer_lstm_state_tuple.h, new_hidden_states)
                new_memory_tapes = update_output(curr_layer, _layer_lstm_state_tuple.c, new_memory_tapes)

                _inputs = _layer_lstm_state_tuple.h

            return _inputs, new_memory_tapes, new_hidden_states, tf.zeros([self._batch_size, self._hidden_size], dtype=tf.float32)

        def _run_with_intra_attention():
            """
            Run with intra-attention
            :return:
                outputs:                    [batch_size, hidden_size]
                hidden_state                [batch_size, num_layer, hidden_dim]
                memory_tape                 [batch_size, num_layer, hidden_dim]
                last_attentive              [batch_size, hidden_size]
            """
            # 1. Perform the first layer
            attentive_hidden_state, attentive_memory_tape = self._calc_intra_attention(
                ts=ts,
                inputs=inputs,
                hidden_states=_get_tape(0, hidden_states),
                memory_tapes=_get_tape(0, memory_tapes),
                last_attentive_hs=last_attentive_hs,
                input_weighted_matrix=self._intra_attention_weight_word
            )

            if not self._is_set_reuse:
                self._scope.reuse_variables()
                self._is_set_reuse = True

            first_layer_hidden_state, first_layer_lstm_state_tuple = tf.nn.dynamic_rnn(
                cell=self._cells[0],
                initial_state=LSTMStateTuple(c=attentive_memory_tape, h=attentive_hidden_state),
                inputs=inputs,
                dtype=tf.float32
            )

            if not self._is_set_reuse:
                self._scope.reuse_variables()
                self._is_set_reuse = True

            new_hidden_states = tf.zeros([self._batch_size, self._num_layer, self._hidden_size], dtype=tf.float32)
            new_memory_tapes = tf.zeros([self._batch_size, self._num_layer, self._hidden_size], dtype=tf.float32)
            new_hidden_states = update_output(0, first_layer_lstm_state_tuple.h, new_hidden_states)
            new_memory_tapes = update_output(0, first_layer_lstm_state_tuple.c, new_memory_tapes)

            # 2. Perform loop, layer by layer

            __last_attentive_hs = attentive_hidden_state
            _inputs = first_layer_lstm_state_tuple.h

            for i, cell in enumerate(self._cells[1:]):
                curr_layer = tf.constant((i + 1))
                ahs, amt = self._calc_intra_attention(
                    ts=ts,
                    inputs=tf.reshape(_inputs, [self._batch_size, 1, self._hidden_size]),
                    hidden_states=_get_tape(curr_layer, hidden_states),
                    memory_tapes=_get_tape(curr_layer, memory_tapes),
                    last_attentive_hs=__last_attentive_hs,
                    input_weighted_matrix=self._intra_attention_weight_l
                )
                _layer_hidden_state, _layer_lstm_state_tuple = tf.nn.dynamic_rnn(
                    cell=cell,
                    initial_state=LSTMStateTuple(c=attentive_memory_tape, h=attentive_hidden_state),
                    inputs=_inputs,
                    dtype=tf.float32
                )
                new_hidden_states = update_output(curr_layer, _layer_lstm_state_tuple.h, new_hidden_states)
                new_memory_tapes = update_output(curr_layer, _layer_lstm_state_tuple.c, new_memory_tapes)

                _inputs = _layer_lstm_state_tuple.h
                __last_attentive_hs = ahs

            self._last_attentive_hidden_state = last_attentive_hs

            return _inputs, new_hidden_states, new_memory_tapes, last_attentive_hs

        _outputs, _new_hidden_states, _new_memory_tapes, _last_attentive_hs = tf.cond(
            tf.logical_and(
                tf.equal(ts, tf.constant(0)),
                tf.not_equal(self._init_states_length, 0)
            ),
            _run_with_init_states,
            _run_with_intra_attention
        )

        return _outputs, _new_hidden_states, _new_memory_tapes, _last_attentive_hs

    def multi_steps(self, inputs):
        """
        Run multi steps
        :param inputs: [batch_size, max_length, embedding_size]
        :return:       [batch_size, max_length, hidden_size]
        """
        values = tf.zeros([self._batch_size, self._max_length, self._hidden_size], dtype=tf.float32)

        hidden_states = tf.zeros(
            [
                self._batch_size,
                self._max_length,
                self._num_layer,
                self._hidden_size
            ],
            dtype=tf.float32
        )
        memory_tapes = tf.zeros(
            [
                self._batch_size,
                self._max_length,
                self._num_layer,
                self._hidden_size
            ]
        )

        last_attentive_hs = tf.zeros([self._batch_size, self._hidden_size], dtype=tf.float32)

        def cond(curr_ts, inputs_embeddings, outputs, _hidden_states, _memory_tapes, _last_attentive_hs):
            return tf.less(curr_ts, self._max_length)

        def _loop_body(curr_ts, inputs_embeddings, outputs, _hidden_states, _memory_tapes, _last_attentive_hs):
            curr_input = tf.reshape(
                tf.slice(inputs_embeddings, [0, curr_ts, 0], [self._batch_size, 1, self._embedding_size]),
                shape=[self._batch_size, 1, self._embedding_size]
            )
            _outputs, _new_hidden_states, _new_memory_tapes, _last_attentive_hs = self.step(
                curr_ts,
                curr_input,
                _hidden_states,
                _memory_tapes,
                _last_attentive_hs
            )
            _outputs = update_output(curr_ts, _outputs, outputs)
            _hidden_states = tf.reshape(
                update_tape(curr_ts, _new_hidden_states, _hidden_states),
                shape=[self._batch_size, self._max_length, self._num_layer, self._hidden_size]
            )
            _memory_tapes = tf.reshape(
                update_tape(curr_ts, _new_memory_tapes, _memory_tapes),
                shape=[self._batch_size, self._max_length, self._num_layer, self._hidden_size]
            )

            _ts = tf.add(curr_ts, 1)
            return _ts, inputs_embeddings, _outputs, _hidden_states, _memory_tapes, _last_attentive_hs

        ts, _inputs_embeddings, values, hidden_states, memory_tapes, last_attentive_hs = tf.while_loop(
            cond=cond,
            body=_loop_body,
            loop_vars=[
                tf.constant(0, tf.int32),
                inputs,
                values,
                hidden_states,
                memory_tapes,
                last_attentive_hs
            ],
            name="multi_step"
        )
        return values, hidden_states, memory_tapes

    def reset_ts(self):
        self._ts = tf.constant(0, dtype=tf.int32)


class Model:
    BEAM_MAX_SCORE = 1e10
    BEAM_MIN_SCORE = -1e10

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

        self._beam_size = util.get_value(opts, "beam_size", 5)

        self._uniform_init_min = util.get_value(opts, "params_init_min", -0.08)
        self._uniform_init_max = util.get_value(opts, "params_init_max", 0.08)

        self._use_intra_attention_in_decode = util.get_value(opts, "use_intra_attention_in_decode", False)

        self._optimizer_choice = util.get_value(opts, "optimizer", "rmsprop")

        self._is_test = is_test

        if self._is_test:
            self._build_test_graph()
        else:
            self._build_train_graph()

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

        self._batch_size = tf.shape(self._encoder_inputs)[0]

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
            # self._embedded shape is self._batch_size * self._max_length_encode * self._embedding_dim
            self._encoder_embedded = tf.nn.embedding_lookup(self._encoder_embeddings, self._encoder_inputs)

        # dropout is only available during training
        # define encoder cell
        with tf.variable_scope("encoder_cell") as scope:
            encoder_cells = list()
            for i in range(self._layers):
                with tf.variable_scope("lstm_cell_%d" % i):
                    encoder_cell = LSTMCell(num_units=self._hidden_dim, state_is_tuple=True)
                    encoder_cell = tf.contrib.rnn.DropoutWrapper(encoder_cell,
                                                                 input_keep_prob=self._encoder_input_keep_prob,
                                                                 output_keep_prob=self._encoder_output_keep_prob)
                    encoder_cells.append(encoder_cell)

        with tf.variable_scope('encoder') as scope:

            template_tuple = LSTMStateTuple(
                h=tf.zeros([self._batch_size, self._hidden_dim], dtype=tf.float32),
                c=tf.zeros([self._batch_size, self._hidden_dim], dtype=tf.float32)
            )
            init_states = [template_tuple]*self._layers

            _encoder_lstmn = LSTMN(
                cells=encoder_cells,
                max_length=self._max_length_encode,
                batch_size=self._batch_size,
                hidden_size=self._hidden_dim,
                embedding_size=self._embedding_dim,
                initial_states=init_states,
                scope=scope
            )

            self._encoder_outputs, self._encoder_hidden_states, self._encoder_memory_tapes = _encoder_lstmn.multi_steps(
                self._encoder_embedded
            )

            # [num_layer, batch_size, hidden_dim]
            last_hidden_state = tf.transpose(
                tf.reshape(
                    tf.slice(
                        self._encoder_hidden_states,
                        [0, tf.subtract(self._max_length_encode, 1), 0, 0],
                        [self._batch_size, 1, self._layers, self._hidden_dim]
                    ),
                    shape=[self._batch_size, self._layers, self._hidden_dim]
                ),
                perm=[1, 0, 2]
            )
            last_memory_cell = tf.transpose(
                tf.reshape(
                    tf.slice(
                        self._encoder_memory_tapes,
                        [0, tf.subtract(self._max_length_encode, 1), 0, 0],
                        [self._batch_size, 1, self._layers, self._hidden_dim]
                    ),
                    shape=[self._batch_size, self._layers, self._hidden_dim]
                ),
                perm=[1, 0, 2]
            )

            # Construct LSTMStateTuple for decoder
            new_tuples = []
            for (h, c) in zip(tf.unstack(last_hidden_state, axis=0), tf.unstack(last_memory_cell, axis=0)):
                new_tuples.append(
                    LSTMStateTuple(h=h, c=c)
                )
            self._encoder_states = tuple(new_tuples)

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

        if not self._use_intra_attention_in_decode:
            # No intra-attention in decode
            with tf.name_scope("decoder_cell"):
                decoder_cell = LSTMCell(num_units=self._hidden_dim, state_is_tuple=True)
                decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell,
                                                             input_keep_prob=self._decoder_input_keep_prob,
                                                             output_keep_prob=self._decoder_output_keep_prob)
                self._decoder_cell = tf.contrib.rnn.MultiRNNCell([decoder_cell] * self._layers,
                                                                 state_is_tuple=True)

            with tf.variable_scope('decoder'):
                self._decoder_outputs, self._decoder_states = tf.nn.dynamic_rnn(self._decoder_cell,
                                                                                initial_state=self._encoder_states,
                                                                                inputs=self._decoder_embedded,
                                                                                dtype=tf.float32)
        else:
            with tf.variable_scope("decoder_cell") as scope:
                decoder_cells = list()
                for i in range(self._layers):
                    with tf.variable_scope("lstm_cell_%d" % i):
                        decoder_cell = LSTMCell(num_units=self._hidden_dim, state_is_tuple=True)
                        decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell,
                                                                     input_keep_prob=self._encoder_input_keep_prob,
                                                                     output_keep_prob=self._encoder_output_keep_prob)
                        decoder_cells.append(decoder_cell)

            with tf.variable_scope('decoder') as scope:

                _decoder_lstmn = LSTMN(
                    cells=decoder_cells,
                    max_length=self._max_length_decode,
                    batch_size=self._batch_size,
                    hidden_size=self._hidden_dim,
                    embedding_size=self._embedding_dim,
                    initial_states=self._encoder_states,
                    scope=scope
                )

                self._decoder_outputs, self._decoder_hidden_states, self._decoder_memory_tapes = _decoder_lstmn.multi_steps(
                    self._decoder_embedded
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

    def _calculate_inter_attention(self, decoder_hidden_states, num):
        """
        Inter-Attention
        Set up attention layer
        :param:  decoder_hidden_states
        :param:  num
        :return: attentive_outputs
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

    def _decode_without_intra_attention(self):
        """
        Decode with Beam Search
        :param decoder_cells:
        :return:
        """
        with tf.name_scope("decoder_cell"):
            decoder_cell = LSTMCell(num_units=self._hidden_dim, state_is_tuple=True)
            decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell,
                                                         input_keep_prob=self._decoder_input_keep_prob,
                                                         output_keep_prob=self._decoder_output_keep_prob)
            decoder_cells = tf.contrib.rnn.MultiRNNCell([decoder_cell] * self._layers,
                                                        state_is_tuple=True)

        # decoder
        with tf.variable_scope('decoder'):

            """
            Batch_size = 1
            """
            batch_size = 1

            # Setup
            decoder_embedded = tf.nn.embedding_lookup(self._decoder_embeddings, [[VocabManager.GO_TOKEN_ID]])
            decoder_outputs, self._decoder_states = tf.nn.dynamic_rnn(decoder_cells,
                                                                      initial_state=self._encoder_states,
                                                                      inputs=decoder_embedded,
                                                                      dtype=tf.float32)
            attention_decoder_outputs = self._calculate_inter_attention(decoder_outputs, num=1)
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
                template = tf.ones([self._beam_size, self._hidden_dim], dtype=tf.float32)
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
                template = tf.ones([self._beam_size, self._hidden_dim], dtype=tf.float32)
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

                _decoder_outputs, _decoder_states = tf.nn.dynamic_rnn(decoder_cells,
                                                                      initial_state=states,
                                                                      inputs=_decoder_embedded,
                                                                      dtype=tf.float32)
                _attention_decoder_outputs = self._calculate_inter_attention(_decoder_outputs, num=1)

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
                ]
            )

    def _decode_with_intra_attention(self):
        # TODO: Add Beam Search
        """
        Decode With intra-attention
        :param decode_cells:
        :return:
        """
        # Use intra-attention in decode
        with tf.variable_scope("decoder_cell"):
            decoder_cells = list()
            for i in range(self._layers):
                with tf.variable_scope("lstm_cell_%d" % i):
                    decoder_cell = LSTMCell(num_units=self._hidden_dim, state_is_tuple=True)
                    decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell,
                                                                 input_keep_prob=self._encoder_input_keep_prob,
                                                                 output_keep_prob=self._encoder_output_keep_prob)
                    decoder_cells.append(decoder_cell)

        with tf.variable_scope('decoder') as scope:

            test_batch_size = 1

            _decoder_lstmn = LSTMN(
                cells=decoder_cells,
                max_length=self._max_length_decode,
                batch_size=test_batch_size,
                hidden_size=self._hidden_dim,
                embedding_size=self._embedding_dim,
                initial_states=self._encoder_states,
                scope=scope
            )

            def _loop_body(token_id, curr_ts, _predictions, hidden_states, memory_tapes, last_attentive_hs):
                decoder_embedded = tf.nn.embedding_lookup(self._decoder_embeddings, token_id)
                _outputs, _new_hidden_states, _new_memory_tapes, _last_attentive_hs = _decoder_lstmn.step(
                    ts=curr_ts,
                    hidden_states=hidden_states,
                    memory_tapes=memory_tapes,
                    inputs=decoder_embedded,
                    last_attentive_hs=last_attentive_hs
                )

                _hidden_states = tf.reshape(
                    update_tape(curr_ts, _new_hidden_states, hidden_states),
                    shape=[test_batch_size, self._max_length_decode, self._layers, self._hidden_dim]
                )
                _memory_tapes = tf.reshape(
                    update_tape(curr_ts, _new_hidden_states, memory_tapes),
                    shape=[test_batch_size, self._max_length_decode, self._layers, self._hidden_dim]
                )

                attention_decoder_outputs = self._calculate_inter_attention(
                    tf.reshape(_outputs, shape=[test_batch_size, 1, self._hidden_dim]),
                    num=1
               )
                softmax_outputs = self._normalize(attention_decoder_outputs)
                prediction = self._predict(softmax_outputs)
                prediction = tf.reshape(prediction, tf.shape(token_id))

                _predictions = _predictions.write(curr_ts, tf.gather_nd(prediction, [0]))
                return prediction, tf.add(curr_ts, 1), _predictions, _hidden_states, _memory_tapes, _last_attentive_hs

            def _terminate_condition(token_id, curr_ts, predictions, hidden_states, memory_tapes, last_attentive_hs):
                """
                :return:
                """
                return tf.less(curr_ts, self._max_length_decode)

            _last_prediction, time_steps, _predictions, decoder_hidden_states, decoder_memory_tapes, decoder_last_attentive_hs = tf.while_loop(
                _terminate_condition,
                _loop_body,
                [
                    tf.constant([[VocabManager.GO_TOKEN_ID]], dtype=tf.int64),
                    tf.constant(0),
                    tf.TensorArray(dtype=tf.int64, size=self._max_length_decode),
                    tf.zeros([test_batch_size, self._max_length_decode, self._layers, self._hidden_dim], dtype=tf.float32),
                    tf.zeros([test_batch_size, self._max_length_decode, self._layers, self._hidden_dim], dtype=tf.float32),
                    tf.zeros([test_batch_size, self._hidden_dim], dtype=tf.float32)
                ]
            )
            self._predictions = _predictions.stack()

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

        self._attention_decoder_outputs = self._calculate_inter_attention(self._decoder_outputs,
                                                                          num=self._max_length_decode)

        softmax_outputs = self._normalize(self._attention_decoder_outputs)

        self._predictions = self._predict(softmax_outputs)

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
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self._learning_rate, decay=0.95)
            else:
                optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)

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

        if not self._use_intra_attention_in_decode:
            self._decode_without_intra_attention()
        else:
            self._decode_with_intra_attention()

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

        if not self._use_intra_attention_in_decode:
            return self._predictions, self._logprobs, self._mask, feed_dict
        else:
            return self._predictions, feed_dict

    def encode(self, batch):
        if self._is_test:
            feed_dict = self._build_test_feed(batch.encoder_seq)
        else:
            feed_dict = self._build_train_feed(batch)
        return self._encoder_outputs, self._encoder_states, self._predictions, self._loss, feed_dict
