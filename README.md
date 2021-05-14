# RelBERT
The official implementation to reproduce the result in **Distilling Relation Embeddings from Pretrained Language Models**.
Datasets and models used in experiments will be downloaded automatically from each resource. 

## Get Started
```shell script
pip install -e .
``` 

## Run Experiment
- **Prompt Search**  
Run prompt search with AutoPrompt/P-tuning 
```shell script
sh examples/experiments/main/propmt.sh
```

- **Model Training**  
RelBERT training
```shell script
sh examples/experiments/main/train.sh
```

- **Evaluation**  
Evaluate on Analogy Test/Lexical Relation Classification.
```shell script
sh examples/experiments/main/evaluate.sh
```

