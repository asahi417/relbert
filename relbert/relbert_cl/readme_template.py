import os
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
      name: Analogy Questions
      type: token-classification
    dataset:
      name: {dataset_alias}
      type: {dataset_alias}
      args: {dataset_alias}
    metrics:
    - name: F1
      type: f1
      value: {metric['micro/f1']}
    - name: Precision
      type: precision
      value: {metric['micro/precision']}
---
# {model_name}

RelBERT fine-tuned from [{config["model"]}](https://huggingface.co/{config["model"]}) on  
{dataset_link}.
Fine-tuning is done via [RelBERT](https://github.com/asahi417/relbert) library (see the repository for more detail).
It achieves the following results on the relation understanding tasks:
- Analogy Question ([full result](https://huggingface.co/{model_name}/raw/main/analogy.json)):
    - SAT: 
    - BATS:
    - U2:
    - U4:
    - Google: 
- Lexical Relation Classification ([full result](https://huggingface.co/{model_name}/raw/main/classification.json))):
    - BLESS:
    - CogALexV:
    - EVALution:
    - K&H+N:
    - ROOT09:
- Relation Mapping ([full result](https://huggingface.co/{model_name}/raw/main/relation_mapping.json)): 


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
