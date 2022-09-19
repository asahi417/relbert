import json
from glob import glob
from relbert.evaluation.validation_loss import evaluate_validation_loss

for i in glob("relbert_output/models/*info_loob*/epoch_*"):
    print(i)
    out = evaluate_validation_loss(
        validation_data='relbert/conceptnet_high_confidence',
        split='full',
        relbert_ckpt=i
    )
    with open(f'{i}/validation_loss.json', 'w') as f:
        json.dump(out, f)

# for p in ["a" "b" "c" "d" "e"]:
#   for METHOD in "average" "average-no-mask" "mask":
#       for e in range(1, 31):
#           ckpt =