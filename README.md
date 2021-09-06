[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/asahi417/relbert/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/relbert.svg)](https://badge.fury.io/py/relbert)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/relbert.svg)](https://pypi.python.org/pypi/relbert/)
[![PyPI status](https://img.shields.io/pypi/status/relbert.svg)](https://pypi.python.org/pypi/relbert/)

# RelBERT
This is the official implementation of
***Distilling Relation Embeddings from Pre-trained Language Models***
(the camera-ready version of the paper will be soon available!)
which has been accepted by the [**EMNLP 2021 main conference**](https://2021.emnlp.org/).

In the paper, we propose RelBERT, that is the state-of-the-art lexical relation embedding model based on large scale pretrained masked language models.
In this repository, we release a python package `relbert` to work around with RelBERT and its checkpoints via [huggingface modelhub](https://huggingface.co/models) and [gensim](https://radimrehurek.com/gensim/).
In brief, what you can do with the `relbert` are summarized as below:
- **Get a high quality embedding vector** given a pair of word
- **Get similar word pairs (nearest neighbors)** given a pair of word
- **Reproduce the results** of our EMNLP 2021 paper.

## Get Started
```shell
pip install relbert
```

## Play with RelBERT
RelBERT can give you a high-quality relation embedding vector of a word pair. First, you need to define the model class with a RelBERT checkpoint.
```python
from relbert import RelBERT
model = RelBERT('asahi417/relbert-roberta-large')
```
As the model checkpoint, we release following three models on the huggingface modelhub.
- [`asahi417/relbert-roberta-large`](https://huggingface.co/asahi417/relbert-roberta-large): RelBERT based on RoBERTa large with custom prompt (recommended as this is the best model in our experiments).
- [`asahi417/relbert-roberta-large-autoprompt`](https://huggingface.co/asahi417/relbert-roberta-large-autoprompt): RelBERT based on RoBERTa large with AutoPrompt.  
- [`asahi417/relbert-roberta-large-ptuning`](https://huggingface.co/asahi417/relbert-roberta-large-ptuning): RelBERT based on RoBERTa large with P-tuning.

Then you just give a list of word to the model to get the embedding.
```python
# the vector has (1, 1024)
v_tokyo_japan = model.get_embedding([['Tokyo', 'Japan']])
```

Let's run a quick experiment to check the embedding quality. Given candidate lists `['Paris', 'France']`, `['apple', 'fruit']`, and `['London', 'Tokyo']`, the pair which shares
the same relation with the `['Tokyo', 'Japan']` is `['Paris', 'France']`. Would the RelBERT embedding be possible to retain it with simple cosine similarity?  
```python
from relbert import cosine_similarity
v_paris_france, v_apple_fruit, v_london_tokyo = model.get_embedding([['Paris', 'France'], ['apple', 'fruit'], ['London', 'Tokyo']])
cosine_similarity(v_tokyo_japan, v_paris_france)
>>> 0.999
cosine_similarity(v_tokyo_japan, v_apple_fruit)
>>> 0.993
cosine_similarity(v_tokyo_japan, v_london_tokyo)
>>> 0.996
```
Bravo! The similarity between `['Tokyo', 'Japan']` and `['Paris', 'France']` is the highest among the candidates.

### Nearest Neighbours of RelBERT
To get the similar word pairs in terms of the RelBERT embedding, we convert the RelBERT embedding to a gensim model file with a fixed vocabulary.
Specifically, we take the vocabulary of the [RELATIVE embedding](http://josecamachocollados.com/papers/relative_ijcai2019.pdf) that is released as a part of
[Analogy Tool](https://github.com/asahi417/AnalogyTools#relative-embedding), and generate the embedding for all the word pairs with RelBERT (`asahi417/relbert-roberta-large`).
Following the original vocabulary representation, words are joined by `__` and multiple token should be combined by `_` such as `New_york__Tokyo`.

The RelBERT embedding gensim file can be found [here](https://drive.google.com/file/d/1z3UeWALwf6EkujI3oYUCwkrIhMuJFdRA/view?usp=sharing). For example, you can get the nearest neighbours as below.
```python
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('gensim_model.bin', binary=True)
model.most_similar('Tokyo__Japan')
>>>  [('Moscow__Russia', 0.9997282028198242),
      ('Cairo__Egypt', 0.9997045993804932),
      ('Baghdad__Iraq', 0.9997043013572693),
      ('Helsinki__Finland', 0.9996970891952515),
      ('Paris__France', 0.999695897102356),
      ('Damascus__Syria', 0.9996891617774963),
      ('Bangkok__Thailand', 0.9996803998947144),
      ('Madrid__Spain', 0.9996673464775085),
      ('Budapest__Hungary', 0.9996543526649475),
      ('Beijing__China', 0.9996539354324341)]
```




## Reproduce the Experiments
To reproduce the experimental result of our EMNLP 2021 paper, you have to clone the repository.
```shell
git clone https://github.com/asahi417/relbert
cd relbert
pip install .
```
First, you need to compute prompts for AutoPrompt and P-tuning.
```shell
sh ./examples/experiments/main/prompt.sh
```
Then, you can train RelBERT model.
```shell
sh ./examples/experiments/main/train.sh
```
Once models are trained, you can evaluate them.
```shell
sh ./examples/experiments/main/evaluate.sh
```

