# Compute Perplexity over Analogy Questions

- Perplexity Computing Module: See [compute_analogy_perplexity.py](./compute_analogy_perplexity.py).

```python
from lm_perplexity import PPL
scorer = PPL('bert-base-cased')
sentence = ['Red is the color of courage, of a warrior and a martyr.',
                'His father was a tailor and his mother was a midwife.',]
print(scorer.get_perplexity(sentence))
[3.896801383195375, 6.4809147517537955]
```

- Computing Perplexity over Analogy Questions
```shell
python compute_analogy_perplexity.py -m 'distilbert-base-uncased' -d 'sat' -p 'is-to-as' -e 'output/sat_distilbert.json'
```
Analogy dataset can be set by `-d` from `sat`/`bats`/`u2`/`u4`/`google`. The prompt can take one from five different one. See more detail by `python compute_analogy_perplexity.py -h`.