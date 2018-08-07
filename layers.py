import tensorflow as tf
from tensorflow.contrib.rnn import MultiRNNCell, RNNCell
# from tensorflow.nn.rnn_cell import DropoutWrapper
DropoutWrapper = tf.nn.rnn_cell.DropoutWrapper
from params import Params

def encoding(words, chars, word_embs, char_embs, scope="emb_encoding"):
    with tf.variable_scope(scope):
        word_encoding = tf.nn.embedding_lookup(word_embs, words)
        char_encoding = tf.nn.embedding_lookup(char_embs, chars)
        return word_encoding, tf.reshape(char_encoding, (words.shape[0], words.shape[1], -1))

def get_attn_params(attn_size, initializer = tf.truncated_normal_initializer):
    pass

def get_rnn_cell(cell_fn, hidden_size, is_training=True):
    # When testing, we should not dropout the rnn cell
    if is_training:
        return DropoutWrapper(cell_fn(hidden_size), output_keep_prob=1-Params.dropout, dtype=tf.float32)
    else:
        return cell_fn(hidden_size)

def bidirectional_RNN(inputs, inputs_len, cell_fn=tf.contrib.rnn.GRUCell, units=Params.attn_size, layers=1, scope="Bidirectional_RNN", is_training=True):
    with tf.variable_scope(scope):

        if layers > 1:
            cell_fw = MultiRNNCell([get_rnn_cell(cell_fn, units, is_training=is_training) for _ in range(layers)])
            cell_bw = MultiRNNCell([get_rnn_cell(cell_fn, units, is_training=is_training) for _ in range(layers)])
        else:
            pass

        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=inputs_len, dtype=tf.float32)

        # outpus is (batch_size, time_steps, hidden_state)
        return tf.concat(outputs, 2)


class AttentionFlowMatchLayer(object):
    """
    Implements the BiDirectional AttentionFlow Model from https://arxiv.org/abs/1611.01603,
    which computes context-to-question Attention and question-to-context Attention
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def match(self, passage_encodes, question_encodes, passage_len, question_len):
        """
        Match the passage_encodes with question_encodes using Attention Flow Match algorithm.
        """
        with tf.variable_scope("BiDAF"):
            sim_matrix = tf.matmul(passage_encodes, question_encodes, transpose_b=True)
            context2question_attn = tf.matmul(tf.nn.softmax(sim_matrix, -1), question_encodes)

            b = tf.nn.softmax(tf.expand_dims(tf.reduce_max(sim_matrix, 2), 1), -1)
            question2context_attn = tf.tile(tf.matmul(b, passage_encodes), [1, tf.shape(passage_encodes)[1], 1])
            
            return tf.concat([passage_encodes, context2question_attn,
                                passage_encodes * context2question_attn,
                                passage_encodes * question2context_attn], -1)



def attend_pooling(pooling_vectors, ref_vector, hidden_size, scope=None):
    
    with tf.variable_scope(scope or "attend_pooling"):
        U = tf.tanh(tf.contrib.layers.fully_connected(pooling_vectors, num_outputs=hidden_size, activation_fn=None, biases_initializer=None)
                    + tf.contrib.layers.fully_connected(tf.expand_dims(ref_vector, 1), num_outputs=hidden_size, activation_fn=None))

        logits = tf.contrib.layers.fully_connected(U, num_outputs=hidden_size, activation_fn=None)
        scores = tf.nn.softmax(logits, 1)
        pooled_vector = tf.reduce_sum(pooling_vectors * scores, axis=1)
    return pooled_vector

# Pointer network
class PointerNetLSTMCell(tf.contrib.rnn.LSTMCell):
    """
    """
    def __init__(self, num_units, context_to_point):
         super(PointerNetLSTMCell, self).__init__(num_units, state_is_tuple=True)
         self.context_to_point = context_to_point

         self.fc_context = tf.contrib.layers.fully_connected(self.context_to_point, num_outputs=self._num_units, activation_fn=None)

    def __call__(self, inputs, state, scope=None):
        (c_prev, m_prev) = state
        with tf.variable_scope(scope or type(self).__name__):
            U = tf.tanh(self.fc_context
                        + tf.expand_dims(tf.contrib.layers.fully_connected(m_prev, num_outputs=self._num_units, activation_fn=None), 1))
            
            logits = tf.contrib.layers.fully_connected(U, num_outputs=1, activation_fn=None)
            scores = tf.nn.softmax(logits, 1)
            attend_context = tf.reduce_sum(self.context_to_point * scores, axis=1)
            lstm_out, lstm_state = super(PointerNetLSTMCell, self).__call__(attend_context, state)

        return tf.squeeze(scores, -1), lstm_state


class PointerNetDecoder(object):

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
    
    def decode(self, passage_vectors, question_vectors, passage_len):
        with tf.variable_scope("ptr-net_decoder"):
            
            random_attn_vector = tf.Variable(tf.random_normal([1, self.hidden_size]), trainable=True, name="random_attn_vector")
            question_pooling = tf.contrib.layers.fully_connected(
                attend_pooling(question_vectors, random_attn_vector, self.hidden_size),
                num_outputs=self.hidden_size, activation_fn=None)
            
            init_state = tf.contrib.rnn.LSTMStateTuple(question_pooling, question_pooling)
            # init_state = tf.contrib.rnn.LSTMStateTuple(question_pooling)


            with tf.variable_scope("fw"):
                fw_cell = PointerNetLSTMCell(self.hidden_size, passage_vectors)
                print("============>", passage_vectors, init_state, self.hidden_size)
                p1_logits, _ = tf.nn.dynamic_rnn(fw_cell, passage_vectors, sequence_length=passage_len, initial_state=init_state)
            
            with tf.variable_scope("bw"):
                bw_cell = PointerNetLSTMCell(self.hidden_size, passage_vectors)
                p2_logits, _ = tf.nn.dynamic_rnn(bw_cell, passage_vectors, sequence_length=passage_len, initial_state=init_state)
            
            return tf.stack((p1_logits, p2_logits), 1)


def pointer_net(passage, passage_len, question, question_len, cell, params, scope="pointer_network"):
    with tf.variable_scope(scope):
        weights_q, weights_p = params
        initial_state = question_pooling(question, units)
