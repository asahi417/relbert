# RelBERT

## Train
- MASK
```shell script
relbert-train --debug -n -t a
relbert-train --debug -n -t b
relbert-train --debug -n -t c
relbert-train --debug -n -t d
relbert-train --debug -n -t e
relbert-train --debug -n -t f
relbert-train --debug -n -t g
relbert-train --debug -n -t h
relbert-train --debug -n -t i
relbert-train --debug -n -t a -p
relbert-train --debug -n -t b -p
relbert-train --debug -n -t c -p
relbert-train --debug -n -t d -p
relbert-train --debug -n -t e -p
relbert-train --debug -n -t f -p
relbert-train --debug -n -t g -p
relbert-train --debug -n -t h -p
relbert-train --debug -n -t i -p
relbert-train --debug -n -t a -s
relbert-train --debug -n -t b -s
relbert-train --debug -n -t c -s
relbert-train --debug -n -t d -s
relbert-train --debug -n -t e -s
relbert-train --debug -n -t f -s
relbert-train --debug -n -t g -s
relbert-train --debug -n -t h -s
relbert-train --debug -n -t i -s
relbert-train --debug -n -t a -p -s
relbert-train --debug -n -t b -p -s
relbert-train --debug -n -t c -p -s
relbert-train --debug -n -t d -p -s
relbert-train --debug -n -t e -p -s
relbert-train --debug -n -t f -p -s
relbert-train --debug -n -t g -p -s
relbert-train --debug -n -t h -p -s
relbert-train --debug -n -t i -p -s
```

- AVG
```shell script
relbert-train --debug -n -t a --mode average
relbert-train --debug -n -t b --mode average
relbert-train --debug -n -t c --mode average
relbert-train --debug -n -t d --mode average
relbert-train --debug -n -t e --mode average
relbert-train --debug -n -t f --mode average
relbert-train --debug -n -t g --mode average
relbert-train --debug -n -t h --mode average
relbert-train --debug -n -t i --mode average
relbert-train --debug -n -t a -p --mode average
relbert-train --debug -n -t b -p --mode average
relbert-train --debug -n -t c -p --mode average
relbert-train --debug -n -t d -p --mode average
relbert-train --debug -n -t e -p --mode average
relbert-train --debug -n -t f -p --mode average
relbert-train --debug -n -t g -p --mode average
relbert-train --debug -n -t h -p --mode average
relbert-train --debug -n -t i -p --mode average
relbert-train --debug -n -t a -s --mode average
relbert-train --debug -n -t b -s --mode average
relbert-train --debug -n -t c -s --mode average
relbert-train --debug -n -t d -s --mode average
relbert-train --debug -n -t e -s --mode average
relbert-train --debug -n -t f -s --mode average
relbert-train --debug -n -t g -s --mode average
relbert-train --debug -n -t h -s --mode average
relbert-train --debug -n -t i -s --mode average
relbert-train --debug -n -t a -p -s --mode average
relbert-train --debug -n -t b -p -s --mode average
relbert-train --debug -n -t c -p -s --mode average
relbert-train --debug -n -t d -p -s --mode average
relbert-train --debug -n -t e -p -s --mode average
relbert-train --debug -n -t f -p -s --mode average
relbert-train --debug -n -t g -p -s --mode average
relbert-train --debug -n -t h -p -s --mode average
relbert-train --debug -n -t i -p -s --mode average
```

- AVG (NO MASK) 
```shell script
relbert-train --debug -n -t a --mode average_no_mask
relbert-train --debug -n -t b --mode average_no_mask
relbert-train --debug -n -t c --mode average_no_mask
relbert-train --debug -n -t d --mode average_no_mask
relbert-train --debug -n -t e --mode average_no_mask
relbert-train --debug -n -t f --mode average_no_mask
relbert-train --debug -n -t g --mode average_no_mask
relbert-train --debug -n -t h --mode average_no_mask
relbert-train --debug -n -t i --mode average_no_mask

relbert-train --debug -n -t a -s --mode average_no_mask
relbert-train --debug -n -t b -s --mode average_no_mask
relbert-train --debug -n -t c -s --mode average_no_mask
relbert-train --debug -n -t d -s --mode average_no_mask
relbert-train --debug -n -t e -s --mode average_no_mask
relbert-train --debug -n -t f -s --mode average_no_mask
relbert-train --debug -n -t g -s --mode average_no_mask
relbert-train --debug -n -t h -s --mode average_no_mask
relbert-train --debug -n -t i -s --mode average_no_mask

relbert-train --debug -n -t a -p --mode average_no_mask
relbert-train --debug -n -t b -p --mode average_no_mask
relbert-train --debug -n -t c -p --mode average_no_mask
relbert-train --debug -n -t d -p --mode average_no_mask
relbert-train --debug -n -t e -p --mode average_no_mask
relbert-train --debug -n -t f -p --mode average_no_mask
relbert-train --debug -n -t g -p --mode average_no_mask
relbert-train --debug -n -t h -p --mode average_no_mask
relbert-train --debug -n -t i -p --mode average_no_mask

relbert-train --debug -n -t a -p -s --mode average_no_mask
relbert-train --debug -n -t b -p -s --mode average_no_mask
relbert-train --debug -n -t c -p -s --mode average_no_mask
relbert-train --debug -n -t d -p -s --mode average_no_mask
relbert-train --debug -n -t e -p -s --mode average_no_mask
relbert-train --debug -n -t f -p -s --mode average_no_mask
relbert-train --debug -n -t g -p -s --mode average_no_mask
relbert-train --debug -n -t h -p -s --mode average_no_mask
relbert-train --debug -n -t i -p -s --mode average_no_mask
```

- AVG (NO MASK, BATS) 
```shell script
relbert-train --debug -n -t a --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t b --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t c --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t d --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t e --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t f --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t g --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t h --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t i --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t a -s --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t b -s --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t c -s --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t d -s --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t e -s --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t f -s --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t g -s --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t h -s --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t i -s --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t a -p --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t b -p --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t c -p --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t d -p --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t e -p --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t f -p --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t g -p --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t h -p --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t i -p --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t a -p -s --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t b -p -s --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t c -p -s --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t d -p -s --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t e -p -s --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t f -p -s --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t g -p -s --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t h -p -s --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
relbert-train --debug -n -t i -p -s --mode average_no_mask --data bats --export-dir ./ckpt/relbert_bats
```


## Eval

```shell script
relbert-eval --debug -m roberta-large,bert-large-cased -t a --mode average_no_mask
relbert-eval --debug -m roberta-large,bert-large-cased -t b --mode average_no_mask
relbert-eval --debug -m roberta-large,bert-large-cased -t c --mode average_no_mask
relbert-eval --debug -m roberta-large,bert-large-cased -t d --mode average_no_mask
relbert-eval --debug -m roberta-large,bert-large-cased -t e --mode average_no_mask
relbert-eval --debug -m roberta-large,bert-large-cased -t f --mode average_no_mask
relbert-eval --debug -m roberta-large,bert-large-cased -t g --mode average_no_mask
relbert-eval --debug -m roberta-large,bert-large-cased -t h --mode average_no_mask
relbert-eval --debug -m roberta-large,bert-large-cased -t i --mode average_no_mask

relbert-eval --debug -m roberta-large,bert-large-cased -t a --mode average
relbert-eval --debug -m roberta-large,bert-large-cased -t b --mode average
relbert-eval --debug -m roberta-large,bert-large-cased -t c --mode average
relbert-eval --debug -m roberta-large,bert-large-cased -t d --mode average
relbert-eval --debug -m roberta-large,bert-large-cased -t e --mode average
relbert-eval --debug -m roberta-large,bert-large-cased -t f --mode average
relbert-eval --debug -m roberta-large,bert-large-cased -t g --mode average
relbert-eval --debug -m roberta-large,bert-large-cased -t h --mode average
relbert-eval --debug -m roberta-large,bert-large-cased -t i --mode average

```