import tensorflow as tf

class V_NET(object):
    def __init__(self, is_training=True, word_emb=None, char_emb=None):
        self.is_training = is_training

        self.word_embeddings = tf.get_variable("word_emb", initializer=tf.constant(word_emb), trainable=False)
        self.char_embeddings = tf.get_variable("char_emb", initializer=tf.constant(char_emb), trainable=False)
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.global_step = tf.Variable(0, name="global_step", trainable=False)


            self.encode_ids()
            self.params = get_attn_params(Params.attn_size, initializer = tf.)

            self.build_matching()
            self.decode()


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
        
        self.passage_encoding = tf.concat((self.passage_word_encoded, self.passage_char_encoded))
        self.question_encoding = tf.concat((self.question_word_encoded, self.question_char_encoded))
        
        # u_t_Q = BiLSTM_Q(utQ, [etQ, ctQ])
        self.passage_encoding = bidirectional_RNN(
            self.passage_encoding,
            self.passage_word_len,
            layers = Params.num_layers,
            scope = "passage_encoding",
            is_training = self.is_training)
        
        self.question_encoding = bidirectional_RNN(
            self.question_encoding,
            self.question_word_len,
            layers = Params.num_layers,
            scope = "question_encoding",
            is_training = self.is_training)

    def build_matching(self):
        match_layer = AttentionFlowMatchLayer(Params.attn_size)

        self.passage_encoding = match_layer.match_layer(self.passage_encoding, self.question_encoding, self.passage_word_len, self.question_word_len)
        if self.is_training:
            self.passage_encoding = tf.nn.dropout(self.passage_encoding, Params.dropout_keep_prob)
        
        self.passage_encoding = bidirectional_RNN(
            self.passage_encoding,
            self.passage_word_len,
            layers = Params.num_layers,
            scope = "fusion",
            is_training = self.is_training
        )

    def decode(self):
        """
        Answer pointer network as proposed in https://arxiv.org/pdf/1506.03134.pdf.
        """
        # pointer_net()
        with tf.variable_scope("pointer_network"):
            vars[""]

            U = 
        batch_size = tf.shape(self.indices)[0]
        concat_passage_encoding = tf.reshape(self.passage_encoding, [batch_size, -1, 2*Params.attn_size])

        
    def boundary_loss(self):
        # negative log probability
        with tf.variable_scope("boundary_loss"):
            # (batch_size, 2) => (batch_size, 2, max_passage_len) and it indices
            self.indices_prob = tf.one_hot(self.indices, tf.shape(self.passage_words)[1])
            # the log is used in labels(indices)
            self.boundary_loss = -tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(tf.log(self.point_logits + 1e-8) * self.indices_prob, 2), 1))
    
    def content_loss(self):
        with tf.variable_scope("content_loss"):
            