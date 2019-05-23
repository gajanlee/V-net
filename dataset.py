import json
import nltk
#nltk.download('punkt')
import pandas as pd
from tqdm import tqdm

from utils import get_fake_answer_json, padding_sequence

infile = "./dev_v2.1.json"
outfile = "./dev.json"


def prepare_corpus(path, output):
    df = pd.read_json(path)
    with open(output, "w") as outFile, tqdm(total=df.shape[0], desc=f"{path}", ascii=True) as pbar:
        for row in df.iterrows():
            pbar.update(1)
            data = row[1]
            answer = data["answers"][0]; query = data["query"]
            query_id = data["query_id"]; query_type = data["query_type"]
            
            passages = sorted([passage for passage in data["passages"]], 
                            key=lambda passage: passage["is_selected"], 
                            reverse=True)

            all_tokens = []; all_spans = []; all_selects = []
            for passage in passages[:5]:
                tokened_text, spans, fake_answer = get_fake_answer_json(passage["passage_text"], answer)
                all_tokens.append(tokened_text)
                all_selects.append(passage["is_selected"])

                if answer == "No Answer Present.":
                    all_spans.append([[0], [0]])
                else:
                    all_spans.append([ [spans[0]], [spans[1]] ])
            
            sample_obj = {
                "documents": all_tokens,
                "spans": all_spans,
                "selects": all_selects,

                "query": nltk.word_tokenize(query),
                "query_id": query_id,
                
                "answer": nltk.word_tokenize(answer),
                # "well_formed_answers": nltk.word_tokenize(data["wellFormedAnswers"][0]),
            }

            outFile.write(json.dumps(sample_obj) + "\n")


def convert_corpus(input_obj):
    """
    Convert the prepared corpus format to `[[input], [output]]` feeds into model.
    """
    def convert_content_indice(passage, span):
        start_span, end_span = span
        return [0]*start_span + [1]*(end_span-start_span) + [0]*(len(passage)-end_span)
    
    passages = input_obj["documents"] #list(map(padding_sequence, input_obj["documents"]))
    spans = input_obj["spans"]
    question = input_obj["query"]
    selects = input_obj["selects"]
    content_indices = [convert_content_indice(passage, span) 
                for passage, span in zip(passages, spans)]

    return ([passages, question],
            [spans, content_indices, selects],)

if __name__ == "__main__":
    prepare_corpus(infile, outfile)