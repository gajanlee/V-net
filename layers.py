import tensorflow as tf

from tensorflow.contrib.rnn import MultiRNNCell, RNNCell
from tensorflow.nn.rnn_cell import DropoutWrapper

def get_attn_params(attn_size, initializer = tf.truncated_normal_initializer):
    pass

def get_rnn_cell(cell_fn, hidden_size, is_training=True):
    # When testing, we should not dropout the rnn cell
    if is_training:
        return DropoutWrapper(cell_fn(hidden_size), output_keep_prob=1-Params.dropout, dtype=tf.flot32)
    else:
        return cell_fn(hidden_size)

def bidirectional_RNN(inputs, inputs_len, cell_fn=tf.contrib.rnn.GRUcell, units=Params.attn_size, layers=1, scope="Bidirectional_RNN", is_training=True):
    with tf.variable_scope(scope):

        if layers > 1:
            cell_fw = MultiRNNCell([get_rnn_cell(cell_fn, units, is_training=is_training) for _ in range(layers)])
            cell_bw = MultiRNNCell([get_rnn_cell(cell_fn, units, is_training=is_training) for _ in range(layers)])
        else:
            pass

        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_lenght=inputs_len, dtype=tf.float32)

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