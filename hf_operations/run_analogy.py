import os
import shutil
from huggingface_hub import ModelFilter, HfApi


api = HfApi()
filt = ModelFilter(author='relbert')
models = api.list_models(filter=filt)
models_filtered = [i.modelId for i in models if 'feature-extraction' in i.tags and i.modelId.startswith('relbert') and i.modelId != 'relbert/relbert-roberta-large']


for i in models_filtered:
    os.system(f"git clone https://huggingface.co/{i}")
    os.system(f"git clone https://huggingface.co/{i}")
    os.system(f'relbert-eval-analogy --overwrite -d "t_rex_relational_similarity" "conceptnet_relational_similarity" "nell_relational_similarity" -s "test" -m "{i}" -o "{i}/analogy.forward.json" -b 64')
    os.system(f'relbert-eval-analogy --overwrite -d "t_rex_relational_similarity" "conceptnet_relational_similarity" "nell_relational_similarity" -s "test" -m "{i}" -o "{i}/analogy.reverse.json" -b 64 --reverse-pair')
    os.system(f'relbert-eval-analogy --overwrite -d "t_rex_relational_similarity" "conceptnet_relational_similarity" "nell_relational_similarity" -s "test"  -m "{i}" -o "{i}/analogy.bidirection.json" -b 64 --bi-direction-pair')
    os.system(f'relbert-eval-analogy --overwrite -d "t_rex_relational_similarity" "conceptnet_relational_similarity" "nell_relational_similarity" -s "validation" -m "{i}" -o "{i}/analogy.forward.json" -b 64')
    os.system(f'relbert-eval-analogy --overwrite -d "t_rex_relational_similarity" "conceptnet_relational_similarity" "nell_relational_similarity" -s "validation" -m "{i}" -o "{i}/analogy.reverse.json" -b 64 --reverse-pair')
    os.system(f'relbert-eval-analogy --overwrite -d "t_rex_relational_similarity" "conceptnet_relational_similarity" "nell_relational_similarity" -s "validation" -m "{i}" -o "{i}/analogy.bidirection.json" -b 64 --bi-direction-pair')
    os.system(f"cd {i} && git lfs install && git add . && git commit -m 'model update' && git push && cd ../")
    shutil.rmtree(i)  # clean up the cloned repo
