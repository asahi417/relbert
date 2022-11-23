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

For causal language model (eg. GPT and OPT), you need to specify `--is-causal` to switch computation.  
```shell
python compute_analogy_perplexity.py --is-causal -m 'gpt2' -d 'sat' -p 'is-to-as' -e 'output/sat_gpt2.json'
```


## Experiment
```shell
experiment_causal () {
  for DATA in 'sat' 'u2' 'u4'
  do
    for TEMP in 'is-to-what' 'is-to-as' 'rel-same' 'what-is-to' 'she-to-as' 'as-what-same'
    do
      python compute_analogy_perplexity.py --is-causal -m "${1}" -b "${2}" -d "${DATA}" -p "${TEMP}" -e "output/perplexity.${1}.${DATA}.${TEMP}.json"
    done
  done
}

experiment_mlm () {
  for DATA in 'sat' 'u2' 'u4'
  do
    for TEMP in 'is-to-what' 'is-to-as' 'rel-same' 'what-is-to' 'she-to-as' 'as-what-same'
    do
      python compute_analogy_perplexity.py -m "${1}" -b "${2}" -d "${DATA}" -p "${TEMP}" -e "output/perplexity.${1}.${DATA}.${TEMP}.json"
    done
  done
}

experiment_causal "gpt2" 128
experiment_causal "gpt2-large" 128
experiment_mlm "roberta-base" 128
experiment_mlm "roberta-large" 128
```