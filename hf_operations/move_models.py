import os
from pprint import pprint
from huggingface_hub import ModelFilter, HfApi

api = HfApi()
filt = ModelFilter(author='relbert')
models = api.list_models(filter=filt)

models_filtered = [i.modelId for i in models if 'feature-extraction' in i.tags and i.modelId.startswith('relbert') and i.modelId not in ['relbert/relbert-roberta-large', 'relbert/relbert-roberta-base']]
models_filtered = [i for i in models_filtered if 'conceptnet' in i or 't-rex' in i or 'nell' in i]

pprint(sorted(models_filtered))

target = 'research-backup'

for i in models_filtered:
    api.move_repo(from_id=i, to_id=f"{target}/{os.path.basename(i)}", repo_type='model')
