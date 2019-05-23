from keras.utils import Sequence
import numpy as np
from dataset import convert_corpus




class BatchGenerator(Sequence):
    "Generator for Keras"

    def __init__(self):
        
        self.vectors = MagitudeVectors(emdim).load_vectors()

        with open(self.input_file, 'r', encoding="utf-8") as f:
            for sample_count, _ in enumerate(f, 1): pass
        
        self.num_of_batches = sample_count // self.batch_size
        self.indices = np.arange(sample_count)
        self.shuffle = True
        

    def __len__(self):
        return self.num_of_batches

    def __getitem__(self, index):
        "Generate one batch of data"
        
        start_index = (index * self.batch_size)
        end_index = ((index+1) * self.batch_size)

        inds = self.indices[start_index: end_index]

        with open(self.input_file, 'r', encoding="utf-8") as inFile:
            for i, line in enumerate(inFile):
                if i in indices:
                    input, output = convert_corpus(json.loads(line))
                    passages, question = input
                    spans, contentIndices, answerIndice = output

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

        return super().__getitem__(index)