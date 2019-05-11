# V-net
Baidu's Machine Comprehension Question Answering Model Implementation [`V-net`](https://arxiv.org/abs/1805.02220).

# Related papers
* [R-net](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf)

* BiDAF
* [Pointer-Network](https://openreview.net/pdf?id=B1-q5Pqxl)
* [V-net](https://arxiv.org/pdf/1805.02220.pdf)

## Dataset

* [MS-MARCO](http://www.msmarco.org/dataset.aspx)

## Word Embedding

* [Glove.word](http://nlp.stanford.edu/data/glove.840B.300d.zip)
* [Glove.char](https://github.com/minimaxir/char-embeddings/blob/master/glove.840B.300d-char.txt)
  
### Tips
* nltk.download("punkt")
```
If urlopen error, go to https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip .
Unzip it into nltk_data/tokenizers.
```