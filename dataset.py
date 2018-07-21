import json
import nltk
import pandas as pd
from tqdm import tqdm

from utils import get_fake_answer_json

infile = "/media/home/lee/MARCO-dataset/dev_v2.0_well_formed.json"
outfile = "./dev.json"

df = pd.read_json(infile)
with open(outfile, "w") as f, tqdm(total=df.size) as pbar:
    for row in df.iterrows():
        pbar.update(1)
        js = {
            "documents": [get_fake_answer_json(p["passage_text"], row[1]["answers"][0]) 
                            for p in row[1]["passages"] if p["is_selected"]],

            "query": nltk.word_tokenize(row[1]["query"]),
            "query_id": row[1]["query_id"],
            
            "answer": nltk.word_tokenize(row[1]["answers"][0]),
            "well_formed_answers": nltk.word_tokenize(row[1]["wellFormedAnswers"][0]),
        }
        f.write(json.dumps(js) + "\n")

