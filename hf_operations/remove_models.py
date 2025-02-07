from pprint import pprint
from huggingface_hub import ModelFilter, HfApi

api = HfApi()
filt = ModelFilter(author='relbert')
models = api.list_models(filter=filt)
models_filtered = [i.modelId for i in models if 'feature-extraction' in i.tags and i.modelId.startswith('relbert') and i.modelId not in ['relbert/relbert-roberta-large', 'relbert/relbert-roberta-base']]
pprint(sorted(models_filtered))
# input("delete all? >>>")
# for i in models_filtered:
#     api.delete_repo(repo_id=i, repo_type='model')
