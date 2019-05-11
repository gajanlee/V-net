from keras.layers import Bidirectional, LSTM, Input, Embedding, Lambda, Flatten, Dense, Softmax
import keras.backend as K
from keras.activations import sigmoid, relu
from layers import *
import keras
from keras.models import Model


from loss_functions import *





from keras import backend as K
import tensorflow as tf







class V_net:
    
    def __init__(self):
        passages_embedding = Input(shape=(Params.max_passage_count, Params.max_passage_len, Params.embedding_dim), 
                                dtype="float32", name="passages_embedding")
        question_embedding = Input(shape=(Params.max_question_len, Params.embedding_dim),
                                dtype="float32", name="question_embedding")

        encode_layer = Bidirectional(LSTM(Params.embedding_dim, #recurrent_keep_prob=1-Params.encoder_dropout, 
                                        return_sequences=True), name="input_encoder")

        def split_encoding_layer(embedding):
            return [encode_layer(passages_embedding[:, i, :, :]) for i in range(Params.max_passage_count)]
        def split_layer(embedding):
            return [passages_embedding[:, i, :, :]  for i in range(Params.max_passage_count)]
        def stack_encoding_layer(encodings):
            return tf.stack(encodings, axis=1)

        passage_embedding_list = Lambda(split_layer, name="split")(passages_embedding)
        # passages_encodings = Lambda(lambda x: K.map_fn(encode_layer, x), Lambda(split_layer)(passages_embedding))
        #passage_encoding_list = Lambda(split_encoding_layer, name="split_embedding")(passages_embedding)
        # passages_encoding = Lambda(stack_encoding_layer)(list(map(encode_layer, passages_encodings)))
            
        #passages_encoding = Lambda(stack_encoding_layer, name="stack_encoding")(passages_encodings)
        question_encoding = encode_layer(question_embedding)
    
        passage_encoding_list = list(map(Lambda(encode_layer), passage_embedding_list))

        #a, b, c, d, e = passage_encoding_list
        #t = Lambda(score_encoding_layer, name="context_encoding")(a)
        #t = Score_encoding_layer()([question_encoding, a])
        #print(t.shape)
        #Lambda(score_encoding_layer, name="context_encoding")(b)
        passage_context_list = [Score_encoding_layer(name=f"context_encoding{i}")([question_encoding, passage_encoding]) for i, passage_encoding in enumerate(passage_encoding_list)]
        # passage_context_list = list(map(Score_encoding_layer(), zip([question_encoding]*len(passage_encoding_list), passage_encoding_list))  )
        
        #passage_context_list = list(map(Lambda(score_encoding_layer, name="context_encoding"), passage_encoding_list))
        model_passage_layer = Bidirectional(LSTM(Params.embedding_dim, recurrent_dropout=0.2, return_sequences=True), name="passage_modeling")

        passage_modeling_list = [model_passage_layer(passage_context) for passage_context in passage_context_list]
        #passage_modeling_list = list(map(Lambda(model_passage_layer, name="modeling_passage"), passage_context_list))


        """2.2 Answer Boundary Loss"""
        def answer_boundary_layer(input):
            passage_encoding, passage_context, passage_modeling = input

            span_begin_probabilities = SpanBegin(name="span_begin")([passage_context, passage_modeling])     
            span_end_probabilities = SpanEnd(name="span_end")(
                [passage_encoding, passage_context, passage_modeling, span_begin_probabilities]
            )

            span_predict = K.stack([span_begin_probabilities, span_end_probabilities], axis=1)
            return span_predict
            #boundary_loss = negative_avg_log_error(span_truths, span_predict)
        
        passage_spans_list = [Lambda(answer_boundary_layer, name=f"passage_spans{i}")([passage_encoding, passage_context, passage_modeling])
                                for i, (passage_encoding, passage_context, passage_modeling) in 
                                enumerate(zip(passage_encoding_list, passage_context_list, passage_modeling_list))]
        
        boundary_output = Lambda(stack_encoding_layer, name="stack_spans")(passage_spans_list)

        """
        2.3 Content Layer
        """
        def content_layer(passage_modeling):
            relu_encoding = Dense(Params.embedding_dim, activation="relu")(passage_modeling) #Dense(100, name="content_activation", activation="relu")(passage_context)
            indice_probability = Dense(1, activation="sigmoid")(relu_encoding) #Dense(100, name="content_indice", activation="sigmoid")(relu_encoding)
            indice_probability = K.squeeze(indice_probability, axis=-1)
            return indice_probability
        
        # Shape: (None, 200)
        indice_probability_list = list(map(Lambda(content_layer, name="content_indice"), passage_modeling_list))
        
        content_output = Lambda(stack_encoding_layer, name="stack_content_indice")(indice_probability_list)

        answer_encoding_list = [Answer_Encoding_layer(name=f"content_encoding{i}")([passage_embedding, indice_probability])
                                for i, (passage_embedding, indice_probability) in enumerate(zip(passage_embedding_list, indice_probability_list))]

        answer_encodings = Lambda(stack_encoding_layer, name="stack_content_encoding")(answer_encoding_list)

        """
        2.4 Verfify
        """
        # verfication = Dense(100, activation="sigmoid")(K.concatenate([answer_encoding, answer_encoding_hat, answer_encoding * answer_encoding_hat]))
        
        # shape (?, 5)
        


        answer_probability = Answer_Probability_layer(name="answer_probability")(answer_encodings)
        
        verify_output= Dense(5)(answer_probability)
        #print(x)
        #verify_output = Lambda(lambda x: x)(answer_probability)
        print(verify_output.shape)

        model = Model([question_embedding, passages_embedding], [boundary_output, content_output, verify_output])


        model.compile(optimizer = keras.optimizers.Adam(0.001),
                    loss = [negative_avg_log_error, content_loss_function, verify_loss_function],
                    )

        model.summary()


        exit()



















        """2.3 Content Loss"""
        content_losses = []; indice_probabilities = []
        for passage_context_encoding in passage_context_encodings:
            relu_encoding = Dense(100, name="content_activation", activation="relu")(passage_context_encoding)
            indice_probability = Dense(100, name="content_indice", activation="sigmoid")(relu_encoding)
            _content_loss = K.losses.categorical_crossentropy(indice_truths, indice_probability)
            content_losses.append(_content_loss)
            indice_probabilities.append(indice_probability)
        content_loss = K.mean(content_losses)

        answer_encodings = []
        for p_word_embedding, p_char_embedding, indice_probability in zip(
            passages_words_embedding, passages_chars_embedding, indice_probabilities):
            answer_encoding = indice_probability * K.concatenate(p_word_embedding, p_char_embedding)
            answer_encodings.append(answer_encoding)
        # answer_encoding = K.mean(answer_encoding)
        
        exit()
        

        
        passages_encoding = Lambda(stack_encoding_layer)(passages_encoding)
        


        model = keras.Model([passages_embedding, question_embedding], [passages_encoding, question_encoding])
        model.summary()

        exit()


        """2.1 Match Layer, question-aware attention."""
        def score_encoding_layer(input):
            question_encoding, passage_encoding = input
            return K.dot(question_encoding, K.transpose(passage_encoding))

        passage_encodings = Lambda(split_encoding_layer, name="split_passage_encoding")(passages_encodings)
        



        passage_context_encodings = []

        for passage_encoding in passage_encodings:
            score_matrix = K.dot(question_encoding, K.transpose(passage_encoding))
            # Follow https://arxiv.org/abs/1611.01603
            # Reference: https://github.com/ParikhKadam/bidaf-keras/blob/master/bidaf/layers/context_to_query.py
            
            context_to_query_attention = C2QAttention(name='context_to_query_attention')([
            score_matrix, question_encoding])
            query_to_context_attention = Q2CAttention(name='query_to_context_attention')([
            score_matrix, passage_encoding])

            merged_context = MergedContext(name='merged_context')(
                                    [passage_encoding, context_to_query_attention, query_to_context_attention])

            modeled_passage = Bidirectional(LSTM(50, recurrent_dropout=0.2, return_sequence=True), name="passage_context_encoding")
            passage_context_encodings.append(
                (merged_context, modeled_passage)
            )
        


        

        passages_words_embedding = Input(shape=(Params.max_passage_count, Params.max_passage_word_len, Params.word_embed_dim), 
                            dtype="float32", name="passages_words_embedding")
        passages_chars_embedding = Input(shape=(Params.max_passage_count, Params.max_passage_word_len, Params.max_passage_char_len*Params.char_embed_dim),
                            dtype="float32", name="passages_chars_embedding")

        question_words_embedding = Input(shape=(Params.max_passage_word_len, Params.word_embed_dim), 
                            dtype="float32", name="question_words_embedding")
        question_chars_embedding = Input(shape=(Params.max_passage_count, Params.char_embed_dim), dtype="float32")
        span_truths = Input(shape=(2,), dtype="int32")


        """2.1 Question and Passage Encoding individually."""
        passage_encodings = []
        for p_word_embedding, p_char_embedding in zip(
            passages_words_embedding, passages_chars_embedding):
            p_encoding = Bidirectional(LSTM(50, 
                            K.concatenate(p_word_embedding, p_char_embedding)))
            passage_encodings.append(p_encoding)
        question_encoding = Bidirectional(LSTM(50, K.concatenate(question_words_embedding, question_chars_embedding)))

        """2.1 Match Layer, question-aware attention."""
        passage_context_encodings = []

        for passage_encoding in passage_encodings:
            score_matrix = K.dot(question_encoding, K.transpose(passage_encoding))
            # Follow https://arxiv.org/abs/1611.01603
            # Reference: https://github.com/ParikhKadam/bidaf-keras/blob/master/bidaf/layers/context_to_query.py
            
            context_to_query_attention = C2QAttention(name='context_to_query_attention')([
            score_matrix, question_encoding])
            query_to_context_attention = Q2CAttention(name='query_to_context_attention')([
            score_matrix, passage_encoding])

            merged_context = MergedContext(name='merged_context')(
                                    [passage_encoding, context_to_query_attention, query_to_context_attention])

            modeled_passage = Bidirectional(LSTM(50, recurrent_dropout=0.2, return_sequence=True), name="passage_context_encoding")
            passage_context_encodings.append(
                (merged_context, modeled_passage)
            )

        
        """2.2 Answer Boundary Loss"""
        concat_passage_encodings = K.concatenate(passage_encodings)
        concat_passage_contexts = K.concatenate(map(lambda x: x[0], passage_context_encodings))
        concat_modeled_passages = K.concatenate(map(lambda x: x[1], passage_context_encodings))

        span_begin_probabilities = SpanBegin(name="span_begin")([concat_passage_contexts, concat_modeled_passages])     
        span_end_probabilities = SpanEnd(name="span_end")(
            [concat_passage_encodings, concat_passage_contexts, concat_modeled_passages, span_begin_probabilities]
        )

        span_predict = K.stack([span_begin_probabilities, span_end_probabilities], axis=1)
        boundary_loss = negative_avg_log_error(span_truths, span_predict)

        """2.3 Content Loss"""
        content_losses = []; indice_probabilities = []
        for passage_context_encoding in passage_context_encodings:
            relu_encoding = Dense(100, name="content_activation", activation="relu")(passage_context_encoding)
            indice_probability = Dense(100, name="content_indice", activation="sigmoid")(relu_encoding)
            _content_loss = K.losses.categorical_crossentropy(indice_truths, indice_probability)
            content_losses.append(_content_loss)
            indice_probabilities.append(indice_probability)
        content_loss = K.mean(content_losses)

        answer_encodings = []
        for p_word_embedding, p_char_embedding, indice_probability in zip(
            passages_words_embedding, passages_chars_embedding, indice_probabilities):
            answer_encoding = indice_probability * K.concatenate(p_word_embedding, p_char_embedding)
            answer_encodings.append(answer_encoding)
        # answer_encoding = K.mean(answer_encoding)

        
        """2.4 Cross-passage verification loss"""
        score_matrix = K.dot(answer_encoding, answer_encoding)
        for i in range(len(score_matrix)):
            score_matrix[i, i] = 0
        weights = K.exp(score_matrix) / K.sum(score_matrix, axis=1)
        answer_encoding_hat = K.sum(weights * answer_encoding, axis=-1)

        verfication = Dense(100, activation="sigmoid")(K.concatenate([answer_encoding, answer_encoding_hat, answer_encoding * answer_encoding_hat]))
        verfication_loss = K.mean(verfication)


        loss = boundary_loss + beta1 * content_loss + beta2 * verfication_loss

        self.model = keras.Model(inputs=[], outputs=[span_predict, boundary_loss, verfication_loss])
        model.add_loss(loss)

        model.compile(
            optimizer = "adam",
            loss = [None]*3,
        )
        model.summary()

        # https://www.jianshu.com/p/4283c25f2a8c



V_net()