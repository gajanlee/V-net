import nltk

from rouge import Rouge

_rouge = Rouge()
def rouge_score(s1, s2):
    return 0 if not s1 or not s2 else _rouge.calc_score([s1], [s2])

def get_fake_answer_json(text, ref_answer):
    """
    The interface to return assembled document json.

    :param text: str, the original passage text
    :param ref_answer: the original marked answer
    :returns document: json, generated data structure
    """
    tokened_text, tokened_answer = [nltk.word_tokenize(t) for t in [text, ref_answer]]
    spans = _get_answer_spans(tokened_text, tokened_answer)
    return {
        "document": tokened_text,
        "answer_spans": spans,
        "fake_answers": tokened_text[spans[0]: spans[1]+1]
    }

def _get_answer_spans(text, ref_answer):
    """
    Based on Rouge-L Score to get the best answer spans.

    :param text: list of tokens in text 
    :param ref_answer: the human's answer, also tokenized
    :returns max_spans: list of two numbers, marks the start and end position with the max score
    """
    max_score = -1.
    max_spans = [0, len(text)-1]
    for start, _token in enumerate(text):
        if _token not in ref_answer: continue
        for end in range(len(text)-1, start-1, -1):
            _score = rouge_score(text[start: end+1], ref_answer)
            if _score > max_score: 
                max_score = _score
                max_spans = [start, end]

    # Warning: the end pointed character is inclueded in fake answer
    return max_spans
