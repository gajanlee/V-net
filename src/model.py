import keras.backend as K
from keras.layers import Bidirectional, LSTM, Input, Embedding, Lambda, Flatten, Dense, Softmax, TimeDistributed
from keras.activations import sigmoid, relu
from keras.models import Model
from layers import *
from params import Params




class Vnet:
    
    def __init__(self):
        passages_embedding = Input(shape=(Params.max_passage_count, Params.max_passage_len, Params.embedding_dim), 
                                        dtype="float32", name="passages_embedding")
        question_embedding = Input(shape=(Params.max_question_len, Params.embedding_dim),
                                dtype="float32", name="question_embedding")

        encode_layer = Bidirectional(LSTM(Params.embedding_dim, #recurrent_keep_prob=1-Params.encoder_dropout, 
                                    return_sequences=True), name="input_encoder")

        question_encoding = encode_layer(question_embedding)
        passage_encoding = TimeDistributed(encode_layer, name="passage_encoding")(passages_embedding)

        passage_context = TimeDistributed(ContextEncoding(question_encoding))(passage_encoding)

        model_passage_layer = Bidirectional(LSTM(Params.embedding_dim, recurrent_dropout=0.2, 
                                                return_sequences=True), name="passage_modeling")
        passage_modeling = TimeDistributed(model_passage_layer, name="passage_modeling")(passage_context)

        span_begin_probabilities = TimeDistributed(SpanBegin(name='span_begin'))(Concatenate([passage_context, passage_modeling]))

        span_end_representation = SpanEndRepresentation([passage_context, passage_modeling, span_begin_probabilities])
        span_end_representation = TimeDistributed(Bidirectional(LSTM(Params.embedding_dim, return_sequences=True)), name="span_end_lstm")(span_end_representation)
        span_end_probabilities = TimeDistributed(SpanEnd(name="span_end_probability"))(Concatenate([passage_context, span_end_representation]))

        content_indices = TimeDistributed(ContentIndice(name="content_indice"))(passage_modeling)
        answer_encoding = AnswerEncoding([passages_embedding, content_indices])

        answer_probability = AnswerProbability(name="answer_probability")(answer_encoding)
        
        model = Model([question_embedding, passages_embedding], [span_begin_probabilities, span_end_probabilities, content_indices, answer_probability])
        model.summary()

Vnet()