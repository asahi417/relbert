# RelBERT
Relational knowledge distilled BERT.

## Get Started
```shell script
pip install -e .
``` 

## Run Experiment
- Prompt Search 
```shell script
sh examples/experiments/main/propmt.sh
```

- Model Training
```shell script
sh examples/experiments/main/train.sh
```

- Evaluation
```shell script
sh examples/experiments/main/evaluate.sh
```

## TODO
- large batch size
- augmentation for classification loss
- test other LM
- put custom template in the config for model hub
- make autoprompt more efficient
- to add word embedding prediction file to analogy tools
- better way to organize lexical classification (save classifier checkpoint?)
- analogy for bert albert 

