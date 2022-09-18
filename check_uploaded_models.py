from pprint import pprint
from huggingface_hub import ModelFilter, HfApi

api = HfApi()
filt = ModelFilter(author='relbert')
models = api.list_models(filter=filt)
models_filtered = [i.modelId for i in models if 'feature-extraction' in i.tags]
models_ex = [i.modelId for i in models if 'feature-extraction' not in i.tags and 'word_embedding_models' not in i.modelId]

pprint(sorted([i for i in models_filtered if 'nce' in i]))
pprint(sorted([i for i in models_filtered if 'triplet' in i]))
pprint(sorted([i for i in models_filtered if 'loob' in i]))
pprint(sorted(models_ex))

for i in models_ex:
    api.delete_repo(i)
