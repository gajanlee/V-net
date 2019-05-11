from keras.engine.topology import Layer
from keras.layers.advanced_activations import Softmax
from keras import backend as K

import tensorflow as tf


class Score_encoding_layer(Layer):
    def __init__(self, **kwargs):
        super(Score_encoding_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Score_encoding_layer, self).build(input_shape)


    def call(self, input):
        question_encoding, passage_encoding = input
        score_matrix = K.squeeze(K.dot(passage_encoding, tf.transpose(question_encoding, perm=(0, 2, 1))), axis=-1)

        context_to_query_attention = C2QAttention(name='context_to_query_attention')([
                                                    score_matrix, question_encoding])
        query_to_context_attention = Q2CAttention(name='query_to_context_attention')([
        score_matrix, passage_encoding])

        merged_context = MergedContext(name='merged_context')(
                                [passage_encoding, context_to_query_attention, query_to_context_attention])

        # modeled_passage = Bidirectional(LSTM(50, recurrent_dropout=0.2, return_sequences=True), name="passage_context_encoding")(modeled_passage)
        return merged_context
    
    def compute_output_shape(self, input_shape):
        question_shape, passage_shape = input_shape

        return (None, passage_shape[1], 512)

    def get_config(self):
        config = super().get_config()
        return config


class Answer_Probability_layer(Layer):
    def __init__(self, **kwargs):
        super(Answer_Probability_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Answer_Probability_layer, self).build(input_shape)

    def __call__(self, answer_encodings):
        score_matrix = tf.matmul(answer_encodings, tf.transpose(answer_encodings, perm=(0, 2, 1)))
        mask = tf.cast(tf.logical_not(tf.cast(tf.matrix_diag([1] * 5), tf.bool)), tf.float32)
        score_matrix = K.dot(score_matrix, mask)
        weights = Softmax()(score_matrix)
        answer_encoding_hat = tf.matmul(weights, answer_encodings)

        answer_representation = K.concatenate([answer_encodings, answer_encoding_hat, answer_encodings*answer_encoding_hat])
        answer_probability = Dense(1, activation="softmax")(answer_representation)

        return K.squeeze(answer_probability, axis=-1)

    def compute_output_shape(self, input_shape):

        return (None, 5)

    def get_config(self):
        config = super().get_config()
        return config

class Answer_Encoding_layer(Layer):
    def __init__(self, **kwargs):
        super(Answer_Encoding_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Answer_Encoding_layer, self).build(input_shape)

    def call(self, inputs):
        passage_embedding, indice_probability = inputs
        answer_encoding = tf.expand_dims(indice_probability, -1) * passage_embedding
        answer_encoding = tf.reduce_sum(answer_encoding, axis=1)
        return answer_encoding

    def compute_output_shape(self, input_shape):
        passage_embedding_shape, indice_probability_shape = input_shape
        return input_shape

    def get_config(self):
        config = super().get_config()
        return config



class Q2CAttention(Layer):

    def __init__(self, **kwargs):
        super(Q2CAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Q2CAttention, self).build(input_shape)

    def call(self, inputs):
        similarity_matrix, encoded_context = inputs
        max_similarity = K.max(similarity_matrix, axis=-1)
        # by default, axis = -1 in Softmax
        context_to_query_attention = Softmax()(max_similarity)
        weighted_sum = K.sum(K.expand_dims(context_to_query_attention, axis=-1) * encoded_context, -2)
        expanded_weighted_sum = K.expand_dims(weighted_sum, 1)
        num_of_repeatations = K.shape(encoded_context)[1]
        return K.tile(expanded_weighted_sum, [1, num_of_repeatations, 1])

    def compute_output_shape(self, input_shape):
        similarity_matrix_shape, encoded_context_shape = input_shape
        return similarity_matrix_shape[:-1] + encoded_context_shape[-1:]

    def get_config(self):
        config = super().get_config()
        return config

from keras.engine.topology import Layer
from keras.layers.advanced_activations import Softmax
from keras import backend as K


class C2QAttention(Layer):

    def __init__(self, **kwargs):
        super(C2QAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        super(C2QAttention, self).build(input_shape)

    def call(self, inputs):
        similarity_matrix, encoded_question = inputs
        context_to_query_attention = Softmax(axis=-1)(similarity_matrix)
        encoded_question = K.expand_dims(encoded_question, axis=1)
        return K.sum(K.expand_dims(context_to_query_attention, axis=-1) * encoded_question, -2)

    def compute_output_shape(self, input_shape):
        similarity_matrix_shape, encoded_question_shape = input_shape
        return similarity_matrix_shape[:-1] + encoded_question_shape[-1:]

    def get_config(self):
        config = super().get_config()
        return config

from keras.engine.topology import Layer
from keras import backend as K


class MergedContext(Layer):

    def __init__(self, **kwargs):
        super(MergedContext, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MergedContext, self).build(input_shape)

    def call(self, inputs):
        encoded_context, context_to_query_attention, query_to_context_attention = inputs
        element_wise_multiply1 = encoded_context * context_to_query_attention
        element_wise_multiply2 = encoded_context * query_to_context_attention
        concatenated_tensor = K.concatenate(
            [encoded_context, context_to_query_attention, element_wise_multiply1, element_wise_multiply2], axis=-1)
        return concatenated_tensor

    def compute_output_shape(self, input_shape):
        encoded_context_shape, _, _ = input_shape
        return encoded_context_shape[:-1] + (encoded_context_shape[-1] * 4, )

    def get_config(self):
        config = super().get_config()
        return config



from keras.engine.topology import Layer
from keras.layers.advanced_activations import Softmax
from keras.layers import TimeDistributed, Dense
from keras import backend as K


class SpanBegin(Layer):

    def __init__(self, **kwargs):
        super(SpanBegin, self).__init__(**kwargs)

    def build(self, input_shape):
        last_dim = input_shape[0][-1] + input_shape[1][-1]
        input_shape_dense_1 = input_shape[0][:-1] + (last_dim, )
        self.dense_1 = Dense(units=1)
        self.dense_1.build(input_shape_dense_1)
        self.trainable_weights = self.dense_1.trainable_weights
        super(SpanBegin, self).build(input_shape)

    def call(self, inputs):
        merged_context, modeled_passage = inputs
        span_begin_input = K.concatenate([merged_context, modeled_passage])
        span_begin_weights = TimeDistributed(self.dense_1)(span_begin_input)
        span_begin_probabilities = Softmax()(K.squeeze(span_begin_weights, axis=-1))
        return span_begin_probabilities

    def compute_output_shape(self, input_shape):
        merged_context_shape, _ = input_shape
        return merged_context_shape[:-1]

    def get_config(self):
        config = super().get_config()
        return config



from keras.engine.topology import Layer
from keras.layers.advanced_activations import Softmax
from keras.layers import TimeDistributed, Dense, LSTM, Bidirectional
from keras import backend as K


class SpanEnd(Layer):

    def __init__(self, **kwargs):
        super(SpanEnd, self).__init__(**kwargs)

    def build(self, input_shape):
        emdim = input_shape[0][-1] // 2
        input_shape_bilstm_1 = input_shape[0][:-1] + (emdim*14, )
        self.bilstm_1 = Bidirectional(LSTM(emdim, return_sequences=True))
        self.bilstm_1.build(input_shape_bilstm_1)
        input_shape_dense_1 = input_shape[0][:-1] + (emdim*10, )
        self.dense_1 = Dense(units=1)
        self.dense_1.build(input_shape_dense_1)
        self.trainable_weights = self.bilstm_1.trainable_weights + self.dense_1.trainable_weights
        super(SpanEnd, self).build(input_shape)

    def call(self, inputs):
        encoded_passage, merged_context, modeled_passage, span_begin_probabilities = inputs
        weighted_sum = K.sum(K.expand_dims(span_begin_probabilities, axis=-1) * modeled_passage, -2)
        passage_weighted_by_predicted_span = K.expand_dims(weighted_sum, axis=1)
        tile_shape = K.concatenate([[1], [K.shape(encoded_passage)[1]], [1]], axis=0)
        passage_weighted_by_predicted_span = K.tile(passage_weighted_by_predicted_span, tile_shape)
        multiply1 = modeled_passage * passage_weighted_by_predicted_span
        span_end_representation = K.concatenate(
            [merged_context, modeled_passage, passage_weighted_by_predicted_span, multiply1])

        span_end_representation = self.bilstm_1(span_end_representation)

        span_end_input = K.concatenate([merged_context, span_end_representation])

        span_end_weights = TimeDistributed(self.dense_1)(span_end_input)

        span_end_probabilities = Softmax()(K.squeeze(span_end_weights, axis=-1))
        return span_end_probabilities

    def compute_output_shape(self, input_shape):
        _, merged_context_shape, _, _ = input_shape
        return merged_context_shape[:-1]

    def get_config(self):
        config = super().get_config()
        return config





def negative_avg_log_error(y_true, y_pred):

    def sum_of_log_probabilities(true_and_pred):
        y_true, y_pred_start, y_pred_end = true_and_pred

        start_probability = y_pred_start[K.cast(y_true[0], dtype='int32')]
        end_probability = y_pred_end[K.cast(y_true[1], dtype='int32')]
        return K.log(start_probability) + K.log(end_probability)

    y_true = K.squeeze(y_true, axis=1)
    y_pred_start = y_pred[:, 0, :]
    y_pred_end = y_pred[:, 1, :]
    batch_probability_sum = K.map_fn(sum_of_log_probabilities, (y_true, y_pred_start, y_pred_end), dtype='float32')
    return -K.mean(batch_probability_sum, axis=0)