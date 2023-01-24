[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/asahi417/relbert/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/relbert.svg)](https://badge.fury.io/py/relbert)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/relbert.svg)](https://pypi.python.org/pypi/relbert/)
[![PyPI status](https://img.shields.io/pypi/status/relbert.svg)](https://pypi.python.org/pypi/relbert/)

# RelBERT
We release the package `relbert` that includes the official implementation of
***Distilling Relation Embeddings from Pre-trained Language Models*** ([https://aclanthology.org/2021.emnlp-main.712/](https://aclanthology.org/2021.emnlp-main.712/))
that has been accepted by the [**EMNLP 2021 main conference**](https://2021.emnlp.org/)

### What's RelBERT?
RelBERT is a state-of-the-art lexical relation embedding model (i.e. model representing any word pair such as "Paris-France" as a fixed-length vector) based on large-scale pretrained masked language models. RelBERT also establishes a very strong baseline to solve analogies in a zero-shot transfer fashion and even outperform strong few-shot models such as [GPT-3](https://arxiv.org/abs/2005.14165) and [Analogical Proportion (AP)](https://aclanthology.org/2021.acl-long.280/).

|                    |   SAT (full) |   SAT |   U2 |   U4 |   Google |   BATS |
|:-------------------|-------------:|------:|-----:|-----:|---------:|-------:|
| [GloVe](https://nlp.stanford.edu/projects/glove/)              |       48.9 |  47.8 | 46.5 | 39.8 |     96   |   68.7 |
| [FastText](https://fasttext.cc/)           |       49.7 |  47.8 | 43   | 40.7 |     96.6 |   72   |
| [RELATIVE](http://josecamachocollados.com/papers/relative_ijcai2019.pdf)           |       24.9 |  24.6 | 32.5 | 27.1 |     62   |   39   |
| [pair2vec](https://arxiv.org/abs/1810.08854)           |       33.7 |  34.1 | 25.4 | 28.2 |     66.6 |   53.8 |
| [GPT-2 (AP)](https://aclanthology.org/2021.acl-long.280/)           | 41.4 | 35.9 | 41.2 | 44.9 | 80.4 | 63.5 |
| [RoBERTa (AP)](https://aclanthology.org/2021.acl-long.280/)         | 49.6 | 42.4 | 49.1 | 49.1 | 90.8 | 69.7 |
| [GPT-2 (tuned AP)](https://aclanthology.org/2021.acl-long.280/)     | 57.8 | 56.7 | 50.9 | 49.5 | 95.2 | 81.2 |
| [RoBERTa (tuned AP)](https://aclanthology.org/2021.acl-long.280/)   | 55.8 | 53.4 | 58.3 | 57.4 | 93.6 | 78.4 | 
| [GPT3 (zeroshot)](https://arxiv.org/abs/2005.14165)               |     53.7   |  - | - | - |  - | - |
| [GPT3 (fewshot)](https://arxiv.org/abs/2005.14165)               |     65.2   |  - | - | - |  - | - |
| ***RelBERT***          |      ***72.2*** |  ***72.7*** | ***65.8*** | ***65.3*** |     ***94.2*** |   ***79.3*** |

[comment]: <> (| ***RelBERT &#40;triplet&#41;***      |      ***67.9*** |  ***67.7*** | ***68.0*** | ***63.2*** |     ***94.2*** |   ***78.9*** |)
[comment]: <> (| ***RelBERT &#40;nce&#41;***          |      ***72.2*** |  ***72.7*** | ***65.8*** | ***65.3*** |     ***94.2*** |   ***79.3*** |)

We also report the performance of RelBERT universal relation embeddings on lexical relation classification datasets, which reinforces the capability of RelBERT to model relations. 
All datasets are public and available in the following links: [analogy questions](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/analogy_test_dataset.zip), [lexical relation classification](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/lexical_relation_dataset.zip).
Please have a look our paper to know more about RelBERT and [AnalogyTool](https://github.com/asahi417/AnalogyTools) or [AP paper](https://aclanthology.org/2021.acl-long.280/) for more information about the datasets.

### What can we do with `relbert`?
In this repository, we release a python package `relbert` to work around with RelBERT and its checkpoints via [huggingface modelhub](https://huggingface.co/models) and [gensim](https://radimrehurek.com/gensim/).
In brief, what you can do with the `relbert` is summarized as below:
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
model = RelBERT()
```

Then you give a word pair to the model to get the embedding.
```python
# the vector has (1024,)
v_tokyo_japan = model.get_embedding(['Tokyo', 'Japan'])
```

Let's run a quick experiment to check the embedding quality. Given candidate lists `['Paris', 'France']`, `['music', 'pizza']`, and `['London', 'Tokyo']`, the pair which shares
the same relation with the `['Tokyo', 'Japan']` is `['Paris', 'France']`. Would the RelBERT embedding be possible to retain it with simple cosine similarity?  
```python
from relbert import cosine_similarity
v_paris_france, v_music_pizza, v_london_tokyo = model.get_embedding([['Paris', 'France'], ['music', 'pizza'], ['London', 'Tokyo']])
cosine_similarity(v_tokyo_japan, v_paris_france)
>>> 0.999
cosine_similarity(v_tokyo_japan, v_music_pizza)
>>> 0.991
cosine_similarity(v_tokyo_japan, v_london_tokyo)
>>> 0.996
```
Bravo! The distance between `['Tokyo', 'Japan']` and `['Paris', 'France']` is the closest among the candidates.
In fact, this pipeline is how we evaluate the RelBERT on the analogy question.

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


## Citation
If you use any of these resources, please cite the following [paper](https://arxiv.org/abs/2110.15705):
```
@inproceedings{ushio-etal-2021-distilling,
    title = "Distilling Relation Embeddings from Pretrained Language Models",
    author = "Ushio, Asahi  and
      Camacho-Collados, Jose  and
      Schockaert, Steven",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.712",
    doi = "10.18653/v1/2021.emnlp-main.712",
    pages = "9044--9062",
    abstract = "Pre-trained language models have been found to capture a surprisingly rich amount of lexical knowledge, ranging from commonsense properties of everyday concepts to detailed factual knowledge about named entities. Among others, this makes it possible to distill high-quality word vectors from pre-trained language models. However, it is currently unclear to what extent it is possible to distill relation embeddings, i.e. vectors that characterize the relationship between two words. Such relation embeddings are appealing because they can, in principle, encode relational knowledge in a more fine-grained way than is possible with knowledge graphs. To obtain relation embeddings from a pre-trained language model, we encode word pairs using a (manually or automatically generated) prompt, and we fine-tune the language model such that relationally similar word pairs yield similar output vectors. We find that the resulting relation embeddings are highly competitive on analogy (unsupervised) and relation classification (supervised) benchmarks, even without any task-specific fine-tuning. Source code to reproduce our experimental results and the model checkpoints are available in the following repository: https://github.com/asahi417/relbert",
}
```
