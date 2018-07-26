import logging
import os
import numpy as np
import pickle
from tqdm import tqdm
from params import Params

class Embedding:
    def __init__(self):
        self._word2index = {"<unk>": 0}
        self._index2word = ["<unk>"]
        self._word_emb = None

        self._char2index = {"<unk>": 0}
        self._index2char = ["unk"]
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

def load_embeddings():
    if os.path.exists(Params.emb_pickle):
        with open(Params.emb_pickle, "rb") as _embf:
            emb = pickle.load(_embf)
    else:
        emb = Embedding()
        emb.build()
        with open(Params.emb_pickle, "wb") as _embf:
            pickle.dump(emb, _embf)
    return emb

