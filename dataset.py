import json
import nltk
#nltk.download('punkt')
import pandas as pd
from tqdm import tqdm

from utils import get_fake_answer_json

infile = "./dev_v2.1.json"
outfile = "./dev.json"

df = pd.read_json(infile)
with open(outfile, "w") as f, tqdm(total=df.size) as pbar:
    for row in df.iterrows():
        pbar.update(1)
        data = row[1]
        answer = data["answers"][0]; query = data["query"]
        query_id = data["query_id"]; query_type = data["query_type"]
        js = {
            "documents": [get_fake_answer_json(passage["passage_text"], answer) 
                            for passage in data["passages"] if passage["is_selected"]==1],

            "query": nltk.word_tokenize(query),
            "query_id": query_id,
            
            "answer": nltk.word_tokenize(answer),
            # "well_formed_answers": nltk.word_tokenize(data["wellFormedAnswers"][0]),
        }
        f.write(json.dumps(js) + "\n")

