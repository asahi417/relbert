from pprint import pprint
from huggingface_hub import ModelFilter, HfApi

api = HfApi()
filt = ModelFilter(author='relbert')
models = api.list_models(filter=filt)
models_filtered = [i.modelId for i in models if 'feature-extraction' in i.tags]
models_ex = [i.modelId for i in models if 'feature-extraction' not in i.tags and 'word_embedding_models' not in i.modelId]

datasets = ['conceptnet-hc', 'semeval2012-v2', 'semeval2012']
loss = ['triplet', 'nce', 'loob']
for dataset in datasets:
    dataset_else = [i for i in datasets if i != dataset]
    for l in loss:
        print(f"## MODEL: {dataset}, LOSS: {l}")
        pprint(sorted([i for i in models_filtered if l in i and dataset in i and not any(i in models_filtered for i in dataset_else)]))
        print()
# pprint(sorted([i for i in models_filtered if 'nce' in i]))
# pprint(sorted([i for i in models_filtered if 'triplet' in i]))
# pprint(sorted([i for i in models_filtered if 'loob' in i]))
# pprint(sorted(models_ex))
#
# for i in models_ex:
#     api.delete_repo(i)
