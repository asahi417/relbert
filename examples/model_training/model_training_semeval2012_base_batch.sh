# NCE LOSS
train_nce () {
  TEMPLATE_ID=${1}
  TEMPLATE=${2}
  BATCH=${3}
  MODEL_CKPT="relbert_output/ckpt/batch_${BATCH}/template-${TEMPLATE_ID}"
  # train
  relbert-train -m "roberta-base" -p -a -o "${MODEL_CKPT}" -b "${BATCH}" -e 10 --loss nce -r 0.000005 -t "${TEMPLATE}"
  # for evaluation
  relbert-eval-analogy -d 'scan' 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64
  relbert-eval-classification -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/classification.json" -b 64
}

for B in 2 4 8 16
do
  train_nce "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>" "${B}"
done
