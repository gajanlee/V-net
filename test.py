from keras.utils import Sequence
import numpy as np
from dataset import convert_corpus
import json

from src.params import Params

Params.embedding_dim = 200

import os
from pymagnitude import Magnitude, MagnitudeUtils


class MagnitudeVectors():

    def __init__(self, emdim=500):

        #base_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
        base_dir = r"C:/Users/rusongli/Desktop/V-net/"
        self.fasttext_dim = 300
        self.glove_dim = emdim - 300

        assert self.glove_dim in [50, 100, 200,
                                  300], "Embedding dimension must be one of the following: 350, 400, 500, 600"

        print("Will download magnitude files from the server if they aren't avaialble locally.. So, grab a cup of coffee while the downloading is under progress..")
        #glove = Magnitude(MagnitudeUtils.download_model('glove/medium/glove.6B.{}d'.format(self.glove_dim),
        #                                                download_dir=os.path.join(base_dir, 'magnitude')), case_insensitive=True)
        #fasttext = Magnitude(MagnitudeUtils.download_model('fasttext/medium/wiki-news-300d-1M-subword',
        #                                                   download_dir=os.path.join(base_dir, 'magnitude')), case_insensitive=True)
        glove = Magnitude(base_dir + "glove.6B.200d.magnitude")
        #fasttext = Magnitude(base_dir + "wiki-news-300d-1M-subword.magnitude")
        
        self.vectors = glove
        #self.vectors = Magnitude(glove, fasttext)

    def load_vectors(self):
        return self.vectors
    

def convert_corpus(input_obj):
    """
    Convert the prepared corpus format to `[[input], [output]]` feeds into model.
    """
    def convert_content_indice(passage, span):
        start_span, end_span = span
        start_span, end_span = start_span[0], end_span[0]
        return [0]*start_span + [1]*(end_span-start_span) + [0]*(len(passage)-end_span) + [0]*(Params.max_passage_len-len(passage))
    
    passages = input_obj["documents"] #list(map(padding_sequence, input_obj["documents"]))
    spans = input_obj["spans"]
    question = input_obj["query"]
    selects = input_obj["selects"]
    content_indices = [convert_content_indice(passage, span) 
                for passage, span in zip(passages, spans)]

    return ([passages, question],
            [spans, content_indices, selects],)


class BatchGenerator(Sequence):
    "Generator for Keras"

    def __init__(self):
        
        self.input_file = "./dev.json"
        self.batch_size = 4
        #self.vectors = MagitudeVectors(emdim).load_vectors()

        with open(self.input_file, 'r', encoding="utf-8") as f:
            for sample_count, _ in enumerate(f, 1): pass
        
        self.num_of_batches = sample_count // self.batch_size
        self.indices = np.arange(sample_count)
        self.shuffle = True
        
        self.vectors = MagnitudeVectors().load_vectors()

        

    def __len__(self):
        return self.num_of_batches

    def __getitem__(self, index):
        "Generate one batch of data"
        
        
        start_index = (index * self.batch_size)
        end_index = ((index+1) * self.batch_size)

        inds = self.indices[start_index: end_index]
        
        batch_passages = np.zeros((self.batch_size, Params.max_passage_count, Params.max_passage_len, Params.embedding_dim))
        batch_question = np.zeros((self.batch_size, Params.max_question_len, Params.embedding_dim))
        batch_spans = np.zeros((self.batch_size, Params.max_passage_count, 2, 1))
        batch_content_indices = np.zeros((self.batch_size, Params.max_passage_count, Params.max_passage_len))
        batch_answer_indice = np.zeros((self.batch_size, Params.max_passage_count))

        
        batch_i = 0
        with open(self.input_file, 'r', encoding="utf-8") as inFile:
            for i, line in enumerate(inFile):
                if i in inds:
                    input, output = convert_corpus(json.loads(line))
                    passages, question = input
                    spans, content_indices, answer_indice = output
                    filter_words = lambda words: list(filter(lambda word: word in self.vectors, words))
                    passages = list(map(filter_words, passages))
                    question = filter_words(question)
                    
                    passage_embedding = self.vectors.query(passages, pad_to_length=Params.max_passage_len)
                    passage_count, passage_len, _ = passage_embedding.shape
                    batch_passages[batch_i, :passage_count, :passage_len, :] = passage_embedding
                    question_embedding = self.vectors.query(question, pad_to_length=Params.max_question_len)
                    question_len, _ = question_embedding.shape
                    batch_question[batch_i, :question_len, :] = question_embedding
                    

                    batch_spans[batch_i, :passage_count, ] = spans
                    batch_content_indices[batch_i, :passage_count, ] = content_indices
                    batch_answer_indice[batch_i, :passage_count, ] = answer_indice
                    
                    batch_i += 1
                    if batch_i == self.batch_size: break
            

        return [batch_passages, batch_question], [batch_spans, batch_content_indices, batch_answer_indice]
        
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


b = BatchGenerator()
from src.model import Vnet
v = Vnet()

m = v.model

from tensorflow.python import debug as tf_debug
import keras.backend as K
import tensorflow as tf
#K.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))
m.fit_generator(b, steps_per_epoch=1)