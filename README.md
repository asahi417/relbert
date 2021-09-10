[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/asahi417/relbert/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/relbert.svg)](https://badge.fury.io/py/relbert)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/relbert.svg)](https://pypi.python.org/pypi/relbert/)
[![PyPI status](https://img.shields.io/pypi/status/relbert.svg)](https://pypi.python.org/pypi/relbert/)

# RelBERT
We release the package `relbert` that includes the official implementation of
***Distilling Relation Embeddings from Pre-trained Language Models***
which has been accepted by the [**EMNLP 2021 main conference**](https://2021.emnlp.org/)
(check the camera-ready version [here](https://github.com/asahi417/relbert/blob/master/asset/EMNLP21_RelBERT_camera.pdf)).

### What's RelBERT?
RelBERT is the state-of-the-art lexical relation embedding model based on large scale pretrained masked language models that establishes a very strong baseline 
in analogy question in zeroshot transfer and even outperform fewshot models such as [GPT-3](https://arxiv.org/abs/2005.14165) and [Analogical Proportion (AP)](https://aclanthology.org/2021.acl-long.280/).

|                    |   SAT (full) |   SAT |   U2 |   U4 |   Google |   BATS |
|:-------------------|-----------:|------:|-----:|-----:|---------:|-------:|
| [GloVe](https://nlp.stanford.edu/projects/glove/)              |       48.9 |  47.8 | 46.5 | 39.8 |     96   |   68.7 |
| [FastText](https://fasttext.cc/)           |       49.7 |  47.8 | 43   | 40.7 |     96.6 |   72   |
| [RELATIVE](http://josecamachocollados.com/papers/relative_ijcai2019.pdf)           |       24.9 |  24.6 | 32.5 | 27.1 |     62   |   39   |
| [pair2vec](https://arxiv.org/abs/1810.08854)           |       33.7 |  34.1 | 25.4 | 28.2 |     66.6 |   53.8 |
| [GPT-2 (AP)](https://aclanthology.org/2021.acl-long.280/)           | 41.4 | 35.9 | 41.2 | 44.9 | 80.4 | 63.5 |
| [RoBERTa (AP)](https://aclanthology.org/2021.acl-long.280/)         | 49.6 | 42.4 | 49.1 | 49.1 | 90.8 | 69.7 |
| [GPT-2 (tuned AP)](https://aclanthology.org/2021.acl-long.280/)     | 57.8 | 56.7 | 50.9 | 49.5 | 95.2 | 81.2 |
| [RoBERTa (tuned AP)](https://aclanthology.org/2021.acl-long.280/)   | 55.8 | 53.4 | 58.3 | 57.4 | 93.6 | 78.4 | 
| [GPT3 (zeroshot)](https://arxiv.org/abs/2005.14165)               |     53.7   |  - | - | - |  - | - |
| [GPT3 (fewshot)](https://arxiv.org/abs/2005.14165)               |     53.7   |  - | - | - |  - | - |
| ***RelBERT***      |      *69.5* |  *70.6* | *66.2* | *65.3* |     *92.4* |   *78.8* |

Please have a look our paper to know more about RelBERT and [AnalogyTool](https://github.com/asahi417/AnalogyTools) or [AP paper](https://aclanthology.org/2021.acl-long.280/) for more information about the analogy question datasets.

### What can we do with `relbert`?
In this repository, we release a python package `relbert` to work around with RelBERT and its checkpoints via [huggingface modelhub](https://huggingface.co/models) and [gensim](https://radimrehurek.com/gensim/).
In brief, what you can do with the `relbert` are summarized as below:
- **Get a high quality embedding vector** given a pair of word
- **Get similar word pairs (nearest neighbors)**
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

Then you give a list of word to the model to get the embedding.
```python
# the vector has (1024,)
v_tokyo_japan = model.get_embedding(['Tokyo', 'Japan'])
```

Let's run a quick experiment to check the embedding quality. Given candidate lists `['Paris', 'France']`, `['music', 'pizza']`, and `['London', 'Tokyo']`, the pair which shares
the same relation with the `['Tokyo', 'Japan']` is `['Paris', 'France']`. Would the RelBERT embedding be possible to retain it with simple cosine similarity?  
```python
from relbert import euclidean_distance
v_paris_france, v_music_pizza, v_london_tokyo = model.get_embedding([['Paris', 'France'], ['music', 'pizza'], ['London', 'Tokyo']])
euclidean_distance(v_tokyo_japan, v_paris_france)
>>> 18.8
euclidean_distance(v_tokyo_japan, v_music_pizza)
>>> 100.7
euclidean_distance(v_tokyo_japan, v_london_tokyo)
>>> 67.8
```
Bravo! The distance between `['Tokyo', 'Japan']` and `['Paris', 'France']` is the closest among the candidates.

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

## Citation
If you use any of these resources, please cite the following paper:
```
@inproceedings{ushio-etal-2021-distilling-relation-embeddings,
    title = "{D}istilling {R}elation {E}mbeddings from {P}re-trained {L}anguage {M}odels",
    author = "Ushio, Asahi  and
      Schockaert, Steven  and
      Camacho-Collados, Jose",
    booktitle = "EMNLP 2021",
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
}
```
