import keras.backend as K
from keras.layers import Bidirectional, LSTM, Input, Embedding, Lambda, Flatten, Dense, Softmax, TimeDistributed, RepeatVector
from keras.activations import sigmoid, relu
from keras.optimizers import Adam, SGD
from keras.models import Model
from .layers import *
from .params import Params
from .loss_functions import *




class Vnet:
    
    def __init__(self):
        passages_embedding = Input(shape=(Params.max_passage_count, Params.max_passage_len, Params.embedding_dim), 
                                        dtype="float32", name="passages_embedding")
        question_embedding = Input(shape=(Params.max_question_len, Params.embedding_dim),
                                dtype="float32", name="question_embedding")

        encode_layer = Bidirectional(LSTM(Params.embedding_dim, #recurrent_keep_prob=1-Params.encoder_dropout, 
                                    return_sequences=True), name="input_encoder")

        question_encoding = encode_layer(question_embedding)
        
        # shape: (None, Params.max_passage_count, Params.max_passage_len, 2*Params.embedding_dim)
        passage_encoding = TimeDistributed(encode_layer, name="passage_encoding")(passages_embedding)
        # shape: (None, Params.max_passage_count, Params.max_passage_len, 8*Params.embedding_dim)
        # passage_context = Concatenate([passage_encoding, passage_encoding, passage_encoding, passage_encoding])
        #ce = ContextEncoding(question_encoding)
        #temp_question_encoding = RepeatVector(Params.max_passage_count)(question_encoding)

        def repeat(x):
            x = K.expand_dims(x, axis=1)
            x = K.tile(x, [1, Params.max_passage_count, 1, 1])
            return x

        temp_question_encoding = Lambda(repeat)(question_encoding)
        temp_passage_encoding = Lambda(lambda i: K.concatenate(i, axis=-2))([passage_encoding, temp_question_encoding])
        passage_context = TimeDistributed(ContextEncoding(), name="passage_context")(temp_passage_encoding)
        model_passage_layer = Bidirectional(LSTM(Params.embedding_dim, recurrent_dropout=0.2, 
                                                return_sequences=True), name="passage_modeling")
        
        # shape: (None, Params.max_passag_count, Params.max_passage_len, 2*Params.embedding_dim)
        passage_modeling = TimeDistributed(model_passage_layer, name="passage_modeling")(passage_context)

        # shape: (None, Params.max_passage_count, Params.max_passage_len)
        span_begin_probabilities = TimeDistributed(SpanBegin(name='span_begin'))(Concatenate([passage_context, passage_modeling]))

        span_end_representation = SpanEndRepresentation([passage_context, passage_modeling, span_begin_probabilities])
        span_end_representation = TimeDistributed(Bidirectional(LSTM(Params.embedding_dim, return_sequences=True)), name="span_end_lstm")(span_end_representation)
        # shape: (None, Params.max_passage_count, Params.max_passage_len)
        span_end_probabilities = TimeDistributed(SpanEnd(name="span_end_probability"))(Concatenate([passage_context, span_end_representation]))

        # shape: (None, max_passage_count, 2, max_passage_len)
        span_probabilities = CombineOutput([span_begin_probabilities, span_end_probabilities])

        # shape: (None, Params.max_passage_count, max_passage_len, 2)
        content_indices = TimeDistributed(ContentIndice(name="content_indice"))(passage_modeling)

        # shape: (None, max_passage_count, max_passage_count)
        answer_encoding = AnswerEncoding([passages_embedding, content_indices])
        # shape: (None, max_passage_count)
        answer_probability = AnswerProbability(name="answer_probability")(answer_encoding)
        
        model = Model(inputs=[passages_embedding, question_embedding], 
                    #outputs=[span_probabilities, content_indices, answer_probability])
                    outputs=[span_probabilities, content_indices, answer_probability])
        model.compile(optimizer = SGD(lr=0.01, decay=1e-3, momentum=0.9, nesterov=True),
                    loss=[boundary_loss, content_loss, verify_loss],)
                    # loss=[boundary_loss, content_loss, "categorical_crossentropy"])
        model.summary()

        self.model = model

#Vnet()