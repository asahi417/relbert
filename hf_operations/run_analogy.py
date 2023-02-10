import os
from huggingface_hub import ModelFilter, HfApi

api = HfApi()
filt = ModelFilter(author='relbert')
models = api.list_models(filter=filt)
models_filtered = [i.modelId for i in models if 'feature-extraction' in i.tags and i.modelId.startswith('relbert') and i.modelId != 'relbert/relbert-roberta-large']
target = 'research-backup'

for i in models_filtered:
    os.system(f"git clone https://huggingface.co/{i}")
    os.system(
        f"cd {opt.model_alias} && git lfs install && git add . && git commit -m 'model update' && git push && cd ../")
    shutil.rmtree(opt.model_alias)  # clean up the cloned repo
