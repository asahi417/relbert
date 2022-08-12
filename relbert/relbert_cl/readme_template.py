from typing import Dict

bib = """
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
"""


def get_readme(model_name: str,
               metric_analogy: Dict,
               metric_classification: Dict,
               metric_relation_mapping: Dict,
               config: Dict):
    config_text = "\n".join([f" - {k}: {v}" for k, v in config.items()])
    dataset_link = f"[{config['data']}](https://huggingface.co/datasets/{config['data']})"
    return f"""---
datasets:
- {config["data"]}
model-index:
- name: {model_name}
  results:
  - task:
      name: Relation Mapping
      type: sorting-task
    dataset:
      name: Relation Mapping
      args: relbert/relation_mapping
    metrics:
    - name: Accuracy
      type: accuracy
      value: {metric_relation_mapping['accuracy']}
  - task:
      name: Analogy Questions (SAT full)
      type: multiple-choice-qa
    dataset:
      name: SAT full
      args: relbert/analogy_questions
    metrics:
    - name: Accuracy
      type: accuracy
      value: {metric_analogy['sat_full']}
  - task:
      name: Analogy Questions (SAT)
      type: multiple-choice-qa
    dataset:
      name: SAT
      args: relbert/analogy_questions
    metrics:
    - name: Accuracy
      type: accuracy
      value: {metric_analogy['sat/test']}
  - task:
      name: Analogy Questions (BATS)
      type: multiple-choice-qa
    dataset:
      name: BATS
      args: relbert/analogy_questions
    metrics:
    - name: Accuracy
      type: accuracy
      value: {metric_analogy['abts/test']}
  - task:
      name: Analogy Questions (Google)
      type: multiple-choice-qa
    dataset:
      name: Google
      args: relbert/analogy_questions
    metrics:
    - name: Accuracy
      type: accuracy
      value: {metric_analogy['google/test']}
  - task:
      name: Analogy Questions (U2)
      type: multiple-choice-qa
    dataset:
      name: U2
      args: relbert/analogy_questions
    metrics:
    - name: Accuracy
      type: accuracy
      value: {metric_analogy['u2/test']}
  - task:
      name: Analogy Questions (U4)
      type: multiple-choice-qa
    dataset:
      name: U4
      args: relbert/analogy_questions
    metrics:
    - name: Accuracy
      type: accuracy
      value: {metric_analogy['u4/test']}
  - task:
      name: Lexical Relation Classification (BLESS)
      type: classification
    dataset:
      name: BLESS
      args: relbert/lexical_relation_classification
    metrics:
    - name: F1
      type: f1
      value: {metric_classification["lexical_relation_classification/BLESS"]["test/f1_micro"]}
    - name: F1 (macro)
      type: f1_macro
      value: {metric_classification["lexical_relation_classification/BLESS"]["test/f1_macro"]}
  - task:
      name: Lexical Relation Classification (CogALexV)
      type: classification
    dataset:
      name: CogALexV
      args: relbert/lexical_relation_classification
    metrics:
    - name: F1
      type: f1
      value: {metric_classification["lexical_relation_classification/CogALexV"]["test/f1_micro"]}
    - name: F1 (macro)
      type: f1_macro
      value: {metric_classification["lexical_relation_classification/CogALexV"]["test/f1_macro"]}
  - task:
      name: Lexical Relation Classification (EVALution)
      type: classification
    dataset:
      name: BLESS
      args: relbert/lexical_relation_classification
    metrics:
    - name: F1
      type: f1
      value: {metric_classification["lexical_relation_classification/EVALution"]["test/f1_micro"]}
    - name: F1 (macro)
      type: f1_macro
      value: {metric_classification["lexical_relation_classification/EVALution"]["test/f1_macro"]}
  - task:
      name: Lexical Relation Classification (K&H+N)
      type: classification
    dataset:
      name: K&H+N
      args: relbert/lexical_relation_classification
    metrics:
    - name: F1
      type: f1
      value: {metric_classification["lexical_relation_classification/K&H+N"]["test/f1_micro"]}
    - name: F1 (macro)
      type: f1_macro
      value: {metric_classification["lexical_relation_classification/K&H+N"]["test/f1_macro"]}
  - task:
      name: Lexical Relation Classification (ROOT09)
      type: classification
    dataset:
      name: ROOT09
      args: relbert/lexical_relation_classification
    metrics:
    - name: F1
      type: f1
      value: {metric_classification["lexical_relation_classification/ROOT09"]["test/f1_micro"]}
    - name: F1 (macro)
      type: f1_macro
      value: {metric_classification["lexical_relation_classification/ROOT09"]["test/f1_macro"]}

---
# {model_name}

RelBERT fine-tuned from [{config["model"]}](https://huggingface.co/{config["model"]}) on  
{dataset_link}.
Fine-tuning is done via [RelBERT](https://github.com/asahi417/relbert) library (see the repository for more detail).
It achieves the following results on the relation understanding tasks:
- Analogy Question ([full result](https://huggingface.co/{model_name}/raw/main/analogy.json)):
    - Accuracy on SAT (full): {metric_analogy['sat_full']} 
    - Accuracy on SAT: {metric_analogy['sat/test']}
    - Accuracy on BATS: {metric_analogy['bats/test']}
    - Accuracy on U2: {metric_analogy['u2/test']}
    - Accuracy on U4: {metric_analogy['u4/test']}
    - Accuracy on Google: {metric_analogy['google/test']}
- Lexical Relation Classification ([full result](https://huggingface.co/{model_name}/raw/main/classification.json))):
    - Micro F1 score on BLESS: {metric_classification["lexical_relation_classification/BLESS"]["test/f1_micro"]}
    - Micro F1 score on CogALexV: {metric_classification["lexical_relation_classification/CogALexV"]["test/f1_micro"]}
    - Micro F1 score on EVALution: {metric_classification["lexical_relation_classification/EVALution"]["test/f1_micro"]}
    - Micro F1 score on K&H+N: {metric_classification["lexical_relation_classification/K&H+N"]["test/f1_micro"]}
    - Micro F1 score on ROOT09: {metric_classification["lexical_relation_classification/ROOT09"]["test/f1_micro"]}
- Relation Mapping ([full result](https://huggingface.co/{model_name}/raw/main/relation_mapping.json)):
    - Accuracy on Relation Mapping: {metric_relation_mapping['accuracy']} 


### Usage
This model can be used through the [relbert library](https://github.com/asahi417/relbert). Install the library via pip   
```shell
pip install relbert
```
and activate model as below.
```python
from relbert import RelBERT
model = RelBERT("{model_name}")
vector = model.get_embedding(['Tokyo', 'Japan'])  # shape of (1024, )
```

### Training hyperparameters

The following hyperparameters were used during training:
{config_text}

The full configuration can be found at [fine-tuning parameter file](https://huggingface.co/{model_name}/raw/main/trainer_config.json).

### Reference
If you use any resource from T-NER, please consider to cite our [paper](https://aclanthology.org/2021.eacl-demos.7/).

```
{bib}
```
"""
