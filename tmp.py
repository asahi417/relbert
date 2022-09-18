from huggingface_hub import HfApi
api = HfApi()


for prompt in ['a', 'b', 'c', 'd', 'e']:
    for m in ['mask', 'average', 'average-no-mask']:
        source = f'relbert/roberta-large-semeval2012-{m}-prompt-{prompt}-nce-classification'
        target = f'relbert/roberta-large-semeval2012-{m}-prompt-{prompt}-nce-classification-conceptnet-validated'
        api.move_repo(from_id=source, to_id=target, repo_type='model')

for prompt in ['a', 'b', 'c', 'd', 'e']:
    for m in ['mask', 'average', 'average-no-mask']:
        source = f'relbert/relbert-roberta-large-semeval2012-{m}-prompt-{prompt}-nce'
        target = f'relbert/roberta-large-semeval2012-{m}-prompt-{prompt}-nce'
        api.move_repo(from_id=source, to_id=target, repo_type='model')


for prompt in ['a', 'b', 'c', 'd', 'e']:
    for m in ['mask', 'average', 'average-no-mask']:
        source = f'relbert/relbert-roberta-large-conceptnet-hc-{m}-prompt-{prompt}-nce'
        target = f'relbert/roberta-large-conceptnet-hc-{m}-prompt-{prompt}-nce'
        api.move_repo(from_id=source, to_id=target, repo_type='model')


for prompt in ['a', 'b', 'c', 'd', 'e']:
    for m in ['mask', 'average', 'average-no-mask']:
        source = f'relbert/relbert-roberta-large-semeval2012-v2-{m}-prompt-{prompt}-nce'
        target = f'relbert/roberta-large-semeval2012-v2-{m}-prompt-{prompt}-nce'
        api.move_repo(from_id=source, to_id=target, repo_type='model')


for prompt in ['a', 'b', 'c', 'd', 'e']:
    for m in ['mask', 'average', 'average-no-mask']:
        source = f'relbert/relbert-roberta-large-semeval2012-{m}-prompt-{prompt}-triplet'
        target = f'relbert/roberta-large-semeval2012-{m}-prompt-{prompt}-triplet'
        api.move_repo(from_id=source, to_id=target, repo_type='model')



for prompt in ['a', 'b', 'c', 'd', 'e']:
    for m in ['mask', 'average', 'average-no-mask']:
        source = f'relbert/relbert-roberta-large-semeval2012-{m}-prompt-{prompt}-loob'
        target = f'relbert/roberta-large-semeval2012-{m}-prompt-{prompt}-loob'
        api.move_repo(from_id=source, to_id=target, repo_type='model')
