import os
import shutil
import json
from tqdm import tqdm

from huggingface_hub import ModelFilter, HfApi
exclude = ['relbert/word_embedding_models', 'relbert/relbert-roberta-large']
api = HfApi()
filt = ModelFilter(author='relbert')
models = api.list_models(filter=filt)
models_filtered = [i.modelId for i in models if 'feature-extraction' in i.tags and i.modelId not in exclude]

for i in tqdm(models_filtered):
    print(i)
    os.system(f"git clone https://huggingface.co/{i}")
    i = os.path.basename(i)
    if os.path.exists(f"{i}/validation_loss.conceptnet_high_confidence.json"):
        os.remove(f"{i}/validation_loss.conceptnet_high_confidence.json")
    if os.path.exists(f"{i}/relation_mapping.json"):
        os.remove(f"{i}/relation_mapping.json")
    with open(f"{i}/validation_loss.json") as f:
        tmp = json.load(f)
        loss = [v for k, v in tmp.items() if k.endswith('loss')][0]
        data = [v for k, v in tmp.items() if k.endswith('data')][0]
        split = list(tmp.keys())[0].split('_')[0]
    new = {
        "loss": loss,
        "data": data,
        "split": split,
        "exclude_relation": None
    }
    with open(f"{i}/validation_loss.json", "w") as f:
        json.dump(new, f)

    os.system(f"cd {i} && git lfs install && git add . && git commit -m 'model update' && git push && cd ../")
    shutil.rmtree(i)

# for prompt in ['a', 'b', 'c', 'd', 'e']:
#     for m in ['mask', 'average', 'average-no-mask']:
#         source = f'relbert/roberta-large-semeval2012-{m}-prompt-{prompt}-nce-classification'
#         target = f'relbert/roberta-large-semeval2012-{m}-prompt-{prompt}-nce-classification-conceptnet-validated'
#         api.move_repo(from_id=source, to_id=target, repo_type='model')
#
# for prompt in ['a', 'b', 'c', 'd', 'e']:
#     for m in ['mask', 'average', 'average-no-mask']:
#         source = f'relbert/relbert-roberta-large-semeval2012-{m}-prompt-{prompt}-nce'
#         target = f'relbert/roberta-large-semeval2012-{m}-prompt-{prompt}-nce'
#         api.move_repo(from_id=source, to_id=target, repo_type='model')
#
#
# for prompt in ['a', 'b', 'c', 'd', 'e']:
#     for m in ['mask', 'average', 'average-no-mask']:
#         source = f'relbert/relbert-roberta-large-conceptnet-hc-{m}-prompt-{prompt}-nce'
#         target = f'relbert/roberta-large-conceptnet-hc-{m}-prompt-{prompt}-nce'
#         api.move_repo(from_id=source, to_id=target, repo_type='model')
#
#
# for prompt in ['a', 'b', 'c', 'd', 'e']:
#     for m in ['mask', 'average', 'average-no-mask']:
#         source = f'relbert/relbert-roberta-large-semeval2012-v2-{m}-prompt-{prompt}-nce'
#         target = f'relbert/roberta-large-semeval2012-v2-{m}-prompt-{prompt}-nce'
#         api.move_repo(from_id=source, to_id=target, repo_type='model')
#
#
# for prompt in ['a', 'b', 'c', 'd', 'e']:
#     for m in ['mask', 'average', 'average-no-mask']:
#         source = f'relbert/relbert-roberta-large-semeval2012-{m}-prompt-{prompt}-triplet'
#         target = f'relbert/roberta-large-semeval2012-{m}-prompt-{prompt}-triplet'
#         api.move_repo(from_id=source, to_id=target, repo_type='model')
#
#
#
# for prompt in ['a', 'b', 'c', 'd', 'e']:
#     for m in ['mask', 'average', 'average-no-mask']:
#         source = f'relbert/relbert-roberta-large-semeval2012-{m}-prompt-{prompt}-loob'
#         target = f'relbert/roberta-large-semeval2012-{m}-prompt-{prompt}-loob'
#         api.move_repo(from_id=source, to_id=target, repo_type='model')
