import keras.backend as K
from keras.engine.topology import Layer
from keras.layers import Dense, Softmax, TimeDistributed, Lambda

import tensorflow as tf

#from .params import Params
class Params:
    max_passage_count = 5
    embedding_dim = 200
    max_passage_len = 200
    max_question_len = 60
    
embedding_dim = 200

def concat(inputs):
    return K.concatenate(inputs)
Concatenate = Lambda(concat, name="concat")




class Similarity:

    def __init__(self, **kwargs):
        super(Similarity, self).__init__(**kwargs)

    def build(self, input_shape):
        pass


class SpanBegin(Layer):
    
    def __init__(self, **kwargs):
        super(SpanBegin, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (None, 200, embeddim*8+embeddim*2)
        self.dense_1 = Dense(units=1)
        self.dense_1.build((input_shape[0], input_shape[-1]))
        self.trainable_weights = self.dense_1.trainable_weights
        super(SpanBegin, self).build(input_shape)

    def call(self, span_begin_input):
        span_begin_weights = TimeDistributed(self.dense_1)(span_begin_input)
        span_begin_probabilities = Softmax()(K.squeeze(span_begin_weights, axis=-1))
        return span_begin_probabilities

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super().get_config()
        return config


class SpanEnd(Layer):
    
    def __init__(self, **kwargs):
        super(SpanEnd, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape_dense_1 = (input_shape[0], embedding_dim*10)
        self.dense_1 = Dense(units=1)
        self.dense_1.build(input_shape_dense_1)
        self.trainable_weights = self.dense_1.trainable_weights
        super(SpanEnd, self).build(input_shape)

    def call(self, span_end_input):
        span_end_weights = TimeDistributed(self.dense_1)(span_end_input)

        span_end_probabilities = Softmax()(K.squeeze(span_end_weights, axis=-1))
        return span_end_probabilities

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super().get_config()
        return config


def span_end_representation(inputs):
    passage_context, passage_modeling, span_begin_probabilities = inputs
    
    weighted_sum = K.sum(K.expand_dims(span_begin_probabilities, axis=-1) * passage_modeling, axis=-2)
    passage_weighted_by_predicted_span = K.expand_dims(weighted_sum, axis=-2)
    passage_weighted_by_predicted_span = K.tile(passage_weighted_by_predicted_span, [1, 1, Params.max_passage_len, 1])
    multiply = passage_modeling * passage_weighted_by_predicted_span
    
    return K.concatenate([passage_context, passage_modeling, passage_weighted_by_predicted_span, multiply])

SpanEndRepresentation = Lambda(span_end_representation, name="span_end_representation")

def combineOutput(inputs):
    span_start_probabilities, span_end_probabilities = inputs
    return K.stack([span_start_probabilities, span_end_probabilities], axis=-2)

CombineOutput = Lambda(combineOutput, name="combineSpanProbability")

class ContentIndice(Layer):
    
    def __init__(self, **kwargs):
        super(ContentIndice, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.dense_1 = Dense(embedding_dim, activation="relu")
        self.dense_1.build(input_shape)
        self.dense_2 = Dense(1, activation="linear")
        self.dense_2.build(input_shape[:-1] + (embedding_dim, ))
        self.trainable_weights = self.dense_1.trainable_weights + self.dense_2.trainable_weights
        
        super(ContentIndice, self).build(input_shape)
        
    def call(self, passage_modeling):
        passage_representation = self.dense_1(passage_modeling)
        passage_representation = self.dense_2(passage_representation)
        passage_representation = K.squeeze(passage_representation, axis=-1)
        # passage_indices = Softmax(axis=-1)(passage_representation)
        return passage_representation
    
    def compute_output_shape(self, input_shape):
        return input_shape[:-1]
    
    def get_config(self):
        config = super().get_config()
        return config


def answerEncoding(inputs):
    passage_embedding, indice_probability = inputs
    answer_encoding = K.expand_dims(indice_probability, axis=-1) * passage_embedding
    answer_encoding = K.sum(answer_encoding, axis=-1)
    return answer_encoding

AnswerEncoding = Lambda(answerEncoding, name="answer_encoding")


class AnswerProbability(Layer):

    def __init__(self, **kwargs):
        super(AnswerProbability, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (None, 5, 200)
        self.dense_1 = Dense(1, activation="relu")
        self.dense_1.build(input_shape[:-1] + (3*input_shape[-1],))
        self.trainable_weights = self.dense_1.trainable_weights
        
        super(AnswerProbability, self).build(input_shape)
    
    def call(self, answer_encoding):
        score_matrix = tf.matmul(answer_encoding, K.permute_dimensions(answer_encoding, (0, 2, 1)))
        eye1 = K.eye(Params.max_passage_count); zero1 = K.zeros_like(eye1); mask = K.cast(K.equal(eye1, zero1), dtype="float32")
        score_matrix = score_matrix * mask
        score_matrix = Softmax(axis=-1)(score_matrix)
        answer_encoding_hat = tf.matmul(score_matrix, answer_encoding)
        answer_encoding_final = K.concatenate([answer_encoding, answer_encoding_hat, answer_encoding*answer_encoding_hat])
        answer_probability = self.dense_1(answer_encoding_final)
        answer_probability = K.squeeze(answer_probability, axis=-1)
        answer_probability = Softmax(axis=-1)(answer_probability)
        return answer_probability

    def compute_output_shape(self, input_shape):
        return (None, input_shape[1])


def slice(x, w1, w2):
    """ Define a tensor slice function
    """
    return x[:, w1:w2, :]


class ContextEncoding(Layer):
    def __init__(self, **kwargs):
        super(ContextEncoding, self).__init__(**kwargs)

    def build(self, input_shape):
        batch_size, p_q_words, embed_dim = input_shape

        self.c2qAttention = C2QAttention(name="context_to_query_attention")
        self.c2qAttention.build(input_shape[-1:] + ())
        self.q2cAttention =  Q2CAttention(name='query_to_context_attention')
        self.q2cAttention.build(input_shape)
        self.mergedContext = MergedContext(name='merged_context')
        self.mergedContext.build(input_shape)

        self.trainable_weights = self.c2qAttention.trainable_weights + self.q2cAttention.trainable_weights + self.mergedContext.trainable_weights
        super(ContextEncoding, self).build(input_shape)


    #def call(self, passage_encoding):
    def call(self, encodings):
        passage_encoding = Lambda(slice, arguments={'w1': 0, 'w2': 200})(encodings)
        question_encoding = Lambda(slice, arguments={'w1': 200, 'w2': 260})(encodings)

        score_matrix = tf.matmul(passage_encoding, K.permute_dimensions(question_encoding, (0, 2, 1)))

        context_to_query_attention = self.c2qAttention([
                                                    score_matrix, question_encoding])
        query_to_context_attention = self.q2cAttention([score_matrix, passage_encoding])

        merged_context = self.mergedContext(
                                [passage_encoding, context_to_query_attention, query_to_context_attention])

        return merged_context
    
    def compute_output_shape(self, input_shape):
        return (None, Params.max_passage_len, 8 * embedding_dim)

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
