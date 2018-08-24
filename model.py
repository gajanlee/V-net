from functools import reduce
import tensorflow as tf
from layers import *
from params import Params
from data_load import *
from utils import *

def log_total_params():
    total_parameters = sum([reduce(lambda x, dim: x*dim.value, variable.get_shape(), 1) for variable in tf.trainable_variables()])
    print("Total number of trainable parameters: {}".format(total_parameters))

class V_NET(object):
    def __init__(self, word_emb=None, char_emb=None):
        self.graph = tf.Graph()

        with self.graph.as_default():
            # embeddings must be inside of the graph
            self.word_embeddings = tf.get_variable("word_emb", shape=word_emb.shape, initializer=tf.constant_initializer(word_emb, dtype=tf.float32), trainable=False)
            self.char_embeddings = tf.get_variable("char_emb", shape=char_emb.shape, initializer=tf.constant_initializer(char_emb, dtype=tf.float32), trainable=False)
            self.global_step = tf.Variable(0, name="global_step", trainable=False)

            self.data, self.num_batch = get_batch(Params.mode)
            (self.passage_words, self.question_words,
            self.passage_chars, self.question_chars,
            self.passage_word_len, self.question_word_len,
            self.passage_char_len, self.question_char_len,
            self.indices, self.words_indices) = self.data

            self.passage_word_len = tf.squeeze(self.passage_word_len, -1)
            self.question_word_len = tf.squeeze(self.question_word_len, -1)

            self.encode_ids()
            self.build_matching()
            self.decode()

            if Params.is_training:
                self.build_train()
                self.summary()

            log_total_params()

    def encode_ids(self):
        self.passage_word_encoded, self.passage_char_encoded = encoding(
             self.passage_words,
             self.passage_chars,
             self.word_embeddings,
             self.char_embeddings,
             scope = "passage_embedding",)
        self.question_word_encoded, self.question_char_encoded = encoding(
             self.question_words,
             self.question_chars,
             self.word_embeddings,
             self.char_embeddings,
             scope = "question_embedding",)
        
        self.passage_encoded = tf.concat((self.passage_word_encoded, self.passage_char_encoded), axis=2)
        self.question_encoded = tf.concat((self.question_word_encoded, self.question_char_encoded), axis=2)
        
        # u_t_Q = BiLSTM_Q(utQ, [etQ, ctQ])
        self.passage_encoding = bidirectional_RNN(
            self.passage_encoded,
            self.passage_word_len,
            layers = Params.num_layers,
            scope = "passage_encoding",
            is_training = Params.is_training)
        
        self.question_encoding = bidirectional_RNN(
            self.question_encoded,
            self.question_word_len,
            layers = Params.num_layers,
            scope = "question_encoding",
            is_training = Params.is_training)
            

    def build_matching(self):
        match_layer = AttentionFlowMatchLayer(Params.attn_size)

        self.passage_encoding = match_layer.match(self.passage_encoding, self.question_encoding, self.passage_word_len, self.question_word_len)
        if Params.is_training:
            self.passage_encoding = tf.nn.dropout(self.passage_encoding, 1-Params.dropout)
        
        self.passage_encoding = bidirectional_RNN(
            self.passage_encoding,
            self.passage_word_len,
            layers = Params.num_layers,
            scope = "fusion",
            is_training = Params.is_training
        )

    def decode(self):
        """
        Answer pointer network as proposed in https://arxiv.org/pdf/1506.03134.pdf.
        """
        # pointer_net()
        with tf.variable_scope("pointer_network"):
            from pointer_net import PointerNetDecoder2
            decoder = PointerNetDecoder2(2*Params.attn_size)     # bidirectional RNN, so it multipy by 2
            self.point_logits = decoder.decode(self.passage_encoding, self.question_encoding)
            # convert point_logits to one matrix
            
            # cell = tf.nn.dropout(tf.contrib.rnn.GRUcell(Params.attn_size * 2), Params.dropout_keep_prob)
            

        
    def build_boundary_loss(self):
        # negative log probability
        with tf.variable_scope("boundary_loss"):
            # (batch_size, 2) => (batch_size, 2, max_passage_len) and it indices
            self.indices_prob = tf.one_hot(self.indices, tf.shape(self.passage_words)[1])
            # the log is used in labels(indices)
            self.boundary_loss = -tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(tf.log(self.point_logits + 1e-8) * self.indices_prob, 2), 1))
    
    def build_content_loss(self):
        with tf.variable_scope("content_loss"):
            # self.word_indices_prob = tf.one_hot(self.word_indices, tf.shape(self.passage_words)[1])

            # the words within the answer spans will be labeled as 1 and others' 0
            self.words_prob = tf.reduce_max(tf.one_hot(self.words_indices, 10), 1)  # how to implement ([[1, 4]]) => ([[0, 1, 1, 1, 1, 0, 0, ...]])            

            self.content_prob = tf.nn.sigmoid(vars["w_1_c"] * tf.nn.relu(vars["w_2_c"]*self.point_logits))
            self.content_loss = -tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(self.words_prob * tf.log(self.content_prob + 1e-8), 2), 1))
        
        self.answer_encoding = tf.reduce_mean(self.content_prob * self.passage_encoded, 2)

    def build_verfication_loss(self):
        
        with tf.variable("verfication_loss"):
            # generate the score matrix, mask the dialog value to zeros.
            mask = tf.cast(tf.logical_not(tf.cast(tf.matrix_diag([1] * Params.max_passage_len), tf.bool)), tf.float32)
            scores = tf.multiply(tf.transpose(self.answer_encoding), self.answer_encoding) * mask   # Warning: Judge if the answer_encoding from the same passage? 
            
            self._answer_encoding = tf.matmul(self.answer_encoding, tf.nn.softmax(scores))
            g_v = tf.contrib.full_connected_layers(tf.concat([self.answer_encoding, self._answer_encoding, self.answer_encoding * self._answer_encoding], 2))

            p_v = tf.nn.softmax(g_v)
            self.cross_passage_verfication = -tf.reduce_mean(tf.log(p_v + 1e-8)*mask, 1)  # y_i is the index of the correct answer
        
        self.loss = self.boundary_loss + Params.beta1 * self.content_loss + Params.beta2 * self.cross_passage_verfication
    
    def build_train(self):
        self.build_boundary_loss()
        self.build_content_loss()
        self.build_verfication_loss()
        self.train_op = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)


def main(_):
    _dict = load_embeddings()    
    model = V_NET(_dict._word_emb, _dict._char_emb)

    # if not os.path.isfile(os.path.join(Params.logdir, "checkpoint")):
        # init = True

    with model.graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sv = tf.train.Supervisor(logdir=Params.logdir, save_model_secs=0, global_step=model.global_step, init_op=tf.global_variables_initializer())

        with sv.managed_session(config=config) as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)

            for epoch in range(1, Params.num_epochs+1):
                if sv.should_stop() or coord.should_stop(): break
                for step in tqdm(range(model.num_batch), total=model.num_batch, ncols=70, leave=False, unit='b'):
                    ls = sess.run(model.train_op)
                    print(ls)
                    if step % Params.save_steps == 0:
                        gs = sess.run(model.global_step)
                        sv.saver.save(sess, Params.logdir + "/model_epoch_%d_step_%d" % (gs/model.num_batch, gs%model.num_batch))

                        # Compute the evaluation scores
                        """sample = np.random.choice(dev_ind, Params.batch_size)
                        feed_dict = {data}
                        index, dev_loss = sess.run([model.output_index, model.loss], feed_dict=feed_dict)

                        Rouge_L = rouge_score()
                        sess.run()"""

            coord.join(threads)
            
if __name__ == "__main__":
    tf.app.run()