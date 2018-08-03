import json, logging, os
import numpy as np
import pickle
import tensorflow as tf
from tqdm import tqdm
from params import Params

"""
    Embedding loader, store data into pickle file.
"""
class Embedding:
    # unknow parameters, its id must be 0
    UNKNOWN, UNKNOWN_ID = "<unk>", 0
    def __init__(self):
        self._word2index = {self.UNKNOWN: self.UNKNOWN_ID}
        self._index2word = [self.UNKNOWN]
        self._word_emb = None

        self._char2index = {self.UNKNOWN: self.UNKNOWN_ID}
        self._index2char = [self.UNKNOWN]
        self._char_emb = None
    
    def build(self):
        self._build(Params.glove_word, self._word2index, self._index2word, "_word_emb", Params.vocab_size, Params.word_emb_size)
        self._build(Params.glove_char, self._char2index, self._index2char, "_char_emb", Params.char_size, Params.char_emb_size)
    
    def _build(self, glove_path, _data2index, _index2data, emb, vocab_size, emb_size):
        _embs = np.zeros((vocab_size, emb_size))
        print(emb)        
        with open(glove_path) as dataf, tqdm(total=vocab_size-1) as pbar:
            for i, line in enumerate(dataf, 1):
                pbar.update(1)
                if i == vocab_size: break
                _d = line.split(" ")
                assert len(_d) == emb_size + 1
                _data2index[_d[0]] = len(_index2data)
                _index2data.append(_d[0])

                _embs[i] = np.asarray(list(map(float, _d[1:])))
                #_embs = np.concatenate((_embs, [list(map(float, _d[1:]))]))
        setattr(self, emb, _embs)

    # Return word/char's id, it supports batch query
    def _get_id(self, query, store):
        if type(query) is str:
            return store[query] if query in store else self.UNKNOWN_ID
        elif type(query) is list:
            return [self._get_id(q, store) for q in query]

    def get_word_id(self, query):
        return self._get_id(query, self._word2index)
    
    def get_char_id(self, query):
        return self._get_id(query, self._char2index)

    def convert_sentences(self, ids):
        return " ".join([self._index2word[id] for id in ids])

# private 
__emb = None
def _load_embeddings():
    if os.path.exists(Params.emb_pickle):
        with open(Params.emb_pickle, "rb") as _embf:
            emb = pickle.load(_embf)
    else:
        emb = Embedding()
        emb.build()
        with open(Params.emb_pickle, "wb") as _embf:
            pickle.dump(emb, _embf)
    return emb

# Lazy init the embedding vocabulary instance,
def load_embeddings():
    global __emb
    if __emb is None:
        __emb = _load_embeddings()
    return __emb

from copy import deepcopy as copy

# padding word data to a standard length, 
# max_len is max_passage(question) word length.
def padding_word(ids, max_len):
    _ids = copy(ids)
    _ids.extend([Embedding.UNKNOWN_ID] * (max_len-len(ids) if max_len>len(ids) else 0))
    return _ids

# max_len is used to mark passage or question,
# the max_char_len is used by Params.
def padding_char(ids, max_len):
    _ids = copy(ids)
    for i, word in enumerate(_ids):
        if len(word) > Params.max_word_len: _ids[i] = [Embedding.UNKNOWN_ID] * Params.max_word_len     # If the word length can't convert to vector, we should mark it as an unknown word
        _ids[i].extend([Embedding.UNKNOWN_ID] * (Params.max_word_len-len(word)))
    _ids.extend([[0] * Params.max_word_len] * (max_len - len(_ids)))
    return _ids

# padding char len is more complex.
def padding_char_len(data, max_len):
    _lens = [min(len(word), Params.max_word_len) for word in data]
    return _lens + [0]*(max_len-len(_lens))

# Every preprocessed json is a corpus data instance
# But, we should decode it into a class to adapt tensorflow batch. 
# class CorpusData:
#     def __init__(self, passage, question, indicies):
#         _dict = load_embeddings()
#         passage_word_ids = padding_word(_dict.get_word_id(document), Params.max_passage_len)
#         question_word_ids = padding_word(_dict.get_word_id(question), Params.max_question_len)
        
#         passage_char_ids = padding_char([_dict.get_char_id(list(pw)) for pw in passage], Params.max_passage_len)
#         question_char_ids = padding_char([_dict.get_char_id(list(qw)) for qw in question], Params.max_question_len)

#         passage_word_len = [len(passage)]
#         question_word_len = [len(question)]
        
#         passage_char_len, question_char_len = [[min(len(word), Params.max_word_len) for word in data] for data in (passage, question)]

#         indicies = indicies
#         self.datas, self.shapes = None, None
"""
    Tensorflow Queue, it can generate batch data.
"""
def get_batch(mode="train"):
    assert getattr(Params, mode + "_path") is not None
    
    data, shapes = load_data(getattr(Params, mode + "_path"))
    
    input_queue = tf.train.slice_input_producer(data, shuffle=False)
    batch = tf.train.batch(input_queue, shapes=shapes, num_threads=2, 
                    batch_size=Params.batch_size, capacity=Params.batch_size*32, dynamic_pad=True)

    return batch, data[0].shape[0] // Params.batch_size


def load_data(path):
    _dict = load_embeddings()

    passage_word_ids, question_word_ids = [], []
    passage_char_ids, question_char_ids = [], []
    passage_word_len, question_word_len = [], []
    passage_char_len, question_char_len = [], []
    indices = []

    with open(path, "rb") as fp:
        for i, line in enumerate(fp):
            if i == 1000: break
            _data = json.loads(line)
            _dict = load_embeddings()

            for _doc in _data["documents"]:
                passage, question = _doc["document"], _data["query"]
                if len(passage) > Params.max_passage_len or len(question) > Params.max_question_len: continue

                passage_word_ids.append( padding_word(_dict.get_word_id(passage), Params.max_passage_len))
                question_word_ids.append( padding_word(_dict.get_word_id(question), Params.max_question_len))
                
                passage_char_ids.append( padding_char([_dict.get_char_id(list(pw)) for pw in passage], Params.max_passage_len))
                question_char_ids.append( padding_char([_dict.get_char_id(list(qw)) for qw in question], Params.max_question_len))

                passage_word_len.append([len(passage)])
                question_word_len.append([len(question)])
                
                passage_char_len.append( padding_char_len(passage, Params.max_passage_len))
                question_char_len.append( padding_char_len(question, Params.max_question_len))

                indices.append([_doc["answer_spans"][0], _doc["answer_spans"][1]+1])
        
    shapes = [(Params.max_passage_len,), (Params.max_question_len, ),
            (Params.max_passage_len, Params.max_word_len, ), (Params.max_question_len, Params.max_word_len, ),
            (1, ), (1, ),
            (Params.max_passage_len, ), (Params.max_question_len, ),
            (2, )]
    
    return ([np.array(passage_word_ids), np.array(question_word_ids),
            np.array(passage_char_ids), np.array(question_char_ids),
            np.array(passage_word_len), np.array(question_word_len),
            np.array(passage_char_len), np.array(question_char_len), np.array(indices)],
            shapes)
