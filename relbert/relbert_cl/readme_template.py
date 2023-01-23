from typing import Dict

bib = """
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
"""


def get_readme(model_name: str,
               config: Dict,
               metric_analogy: Dict = None,
               metric_classification: Dict = None,
               metric_relation_mapping: Dict = None):
    config_text = "\n".join([f' - {k}: {v}' for k, v in config.items() if k != "template"])
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
      type: relation-mapping
    metrics:
    - name: Accuracy
      type: accuracy
      value: {metric_relation_mapping['accuracy'] if metric_relation_mapping is not None else None}
  - task:
      name: Analogy Questions (SAT full)
      type: multiple-choice-qa
    dataset:
      name: SAT full
      args: relbert/analogy_questions
      type: analogy-questions
    metrics:
    - name: Accuracy
      type: accuracy
      value: {metric_analogy['sat_full'] if metric_analogy is not None else None}
  - task:
      name: Analogy Questions (SAT)
      type: multiple-choice-qa
    dataset:
      name: SAT
      args: relbert/analogy_questions
      type: analogy-questions
    metrics:
    - name: Accuracy
      type: accuracy
      value: {metric_analogy['sat/test'] if metric_analogy is not None else None}
  - task:
      name: Analogy Questions (BATS)
      type: multiple-choice-qa
    dataset:
      name: BATS
      args: relbert/analogy_questions
      type: analogy-questions
    metrics:
    - name: Accuracy
      type: accuracy
      value: {metric_analogy['bats/test'] if metric_analogy is not None else None}
  - task:
      name: Analogy Questions (Google)
      type: multiple-choice-qa
    dataset:
      name: Google
      args: relbert/analogy_questions
      type: analogy-questions
    metrics:
    - name: Accuracy
      type: accuracy
      value: {metric_analogy['google/test'] if metric_analogy is not None else None}
  - task:
      name: Analogy Questions (U2)
      type: multiple-choice-qa
    dataset:
      name: U2
      args: relbert/analogy_questions
      type: analogy-questions
    metrics:
    - name: Accuracy
      type: accuracy
      value: {metric_analogy['u2/test'] if metric_analogy is not None else None}
  - task:
      name: Analogy Questions (U4)
      type: multiple-choice-qa
    dataset:
      name: U4
      args: relbert/analogy_questions
      type: analogy-questions
    metrics:
    - name: Accuracy
      type: accuracy
      value: {metric_analogy['u4/test'] if metric_analogy is not None else None}
  - task:
      name: Lexical Relation Classification (BLESS)
      type: classification
    dataset:
      name: BLESS
      args: relbert/lexical_relation_classification
      type: relation-classification
    metrics:
    - name: F1
      type: f1
      value: {metric_classification["lexical_relation_classification/BLESS"]["test/f1_micro"] if metric_classification is not None else None}
    - name: F1 (macro)
      type: f1_macro
      value: {metric_classification["lexical_relation_classification/BLESS"]["test/f1_macro"] if metric_classification is not None else None}
  - task:
      name: Lexical Relation Classification (CogALexV)
      type: classification
    dataset:
      name: CogALexV
      args: relbert/lexical_relation_classification
      type: relation-classification
    metrics:
    - name: F1
      type: f1
      value: {metric_classification["lexical_relation_classification/CogALexV"]["test/f1_micro"] if metric_classification is not None else None}
    - name: F1 (macro)
      type: f1_macro
      value: {metric_classification["lexical_relation_classification/CogALexV"]["test/f1_macro"] if metric_classification is not None else None}
  - task:
      name: Lexical Relation Classification (EVALution)
      type: classification
    dataset:
      name: BLESS
      args: relbert/lexical_relation_classification
      type: relation-classification
    metrics:
    - name: F1
      type: f1
      value: {metric_classification["lexical_relation_classification/EVALution"]["test/f1_micro"] if metric_classification is not None else None}
    - name: F1 (macro)
      type: f1_macro
      value: {metric_classification["lexical_relation_classification/EVALution"]["test/f1_macro"] if metric_classification is not None else None}
  - task:
      name: Lexical Relation Classification (K&H+N)
      type: classification
    dataset:
      name: K&H+N
      args: relbert/lexical_relation_classification
      type: relation-classification
    metrics:
    - name: F1
      type: f1
      value: {metric_classification["lexical_relation_classification/K&H+N"]["test/f1_micro"] if metric_classification is not None else None}
    - name: F1 (macro)
      type: f1_macro
      value: {metric_classification["lexical_relation_classification/K&H+N"]["test/f1_macro"] if metric_classification is not None else None}
  - task:
      name: Lexical Relation Classification (ROOT09)
      type: classification
    dataset:
      name: ROOT09
      args: relbert/lexical_relation_classification
      type: relation-classification
    metrics:
    - name: F1
      type: f1
      value: {metric_classification["lexical_relation_classification/ROOT09"]["test/f1_micro"] if metric_classification is not None else None}
    - name: F1 (macro)
      type: f1_macro
      value: {metric_classification["lexical_relation_classification/ROOT09"]["test/f1_macro"] if metric_classification is not None else None}

---
# {model_name}

RelBERT based on [{config["model"]}](https://huggingface.co/{config["model"]}) fine-tuned on {dataset_link} (see the [`relbert`](https://github.com/asahi417/relbert) for more detail of fine-tuning).
This model achieves the following results on the relation understanding tasks:
- Analogy Question ([dataset](https://huggingface.co/datasets/relbert/analogy_questions), [full result](https://huggingface.co/{model_name}/raw/main/analogy.forward.json)):
    - Accuracy on SAT (full): {metric_analogy['sat_full'] if metric_analogy is not None else None} 
    - Accuracy on SAT: {metric_analogy['sat/test'] if metric_analogy is not None else None}
    - Accuracy on BATS: {metric_analogy['bats/test'] if metric_analogy is not None else None}
    - Accuracy on U2: {metric_analogy['u2/test'] if metric_analogy is not None else None}
    - Accuracy on U4: {metric_analogy['u4/test'] if metric_analogy is not None else None}
    - Accuracy on Google: {metric_analogy['google/test'] if metric_analogy is not None else None}
- Lexical Relation Classification ([dataset](https://huggingface.co/datasets/relbert/lexical_relation_classification), [full result](https://huggingface.co/{model_name}/raw/main/classification.json)):
    - Micro F1 score on BLESS: {metric_classification["lexical_relation_classification/BLESS"]["test/f1_micro"] if metric_classification is not None else None}
    - Micro F1 score on CogALexV: {metric_classification["lexical_relation_classification/CogALexV"]["test/f1_micro"] if metric_classification is not None else None}
    - Micro F1 score on EVALution: {metric_classification["lexical_relation_classification/EVALution"]["test/f1_micro"] if metric_classification is not None else None}
    - Micro F1 score on K&H+N: {metric_classification["lexical_relation_classification/K&H+N"]["test/f1_micro"] if metric_classification is not None else None}
    - Micro F1 score on ROOT09: {metric_classification["lexical_relation_classification/ROOT09"]["test/f1_micro"] if metric_classification is not None else None}
- Relation Mapping ([dataset](https://huggingface.co/datasets/relbert/relation_mapping), [full result](https://huggingface.co/{model_name}/raw/main/relation_mapping.json)):
    - Accuracy on Relation Mapping: {metric_relation_mapping['accuracy'] if metric_relation_mapping is not None else None} 


### Usage
This model can be used through the [relbert library](https://github.com/asahi417/relbert). Install the library via pip   
```shell
pip install relbert
```
and activate model as below.
```python
from relbert import RelBERT
model = RelBERT("{model_name}")
vector = model.get_embedding(['Tokyo', 'Japan'])  # shape of (n_dim, )
```

### Training hyperparameters

{config_text}

See the full configuration at [config file](https://huggingface.co/{model_name}/raw/main/finetuning_config.json).

### Reference
If you use any resource from RelBERT, please consider to cite our [paper](https://aclanthology.org/2021.emnlp-main.712/).

```
{bib}
```
"""