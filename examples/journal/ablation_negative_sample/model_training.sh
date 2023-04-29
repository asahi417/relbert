# NCE LOSS
train_nce () {
  TEMPLATE_ID=${1}
  TEMPLATE=${2}
  SAMPLE=${3}
  MODEL_CKPT="relbert_output/ckpt/negative_sample_${SAMPLE}/template-${TEMPLATE_ID}"
  # train
  relbert-train -m "roberta-base" -p -a -o "${MODEL_CKPT}" -b 32 -e 10 --loss nce -r 0.000005 -t "${TEMPLATE}" --num-negative "${SAMPLE}"
  for E in 1 2 3 4 5 6 7 8 9
  do
    relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/epoch_${E}" -o "${MODEL_CKPT}/epoch_${E}/analogy.forward.json" -b 64
  done
  relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64
}

for N in 25 50 100 150 200 250 300 350
do
  train_nce "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>" "${N}"
  train_nce "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>" "${N}"
  train_nce "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>" "${N}"
  train_nce "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" "${N}"
  train_nce "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>" "${N}"
done


eval_nce() {
  TEMPLATE_ID=${1}
  SAMPLE=${2}
  CKPT=${3}
  MODEL_CKPT="relbert_output/ckpt/negative_sample_${SAMPLE}/template-${TEMPLATE_ID}/${CKPT}"

  # for evaluation
  relbert-eval-analogy -d 'scan' 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.forward.json" -b 64
  relbert-eval-analogy -d 'scan' 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'validation' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.forward.json" -b 64
  relbert-eval-classification -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/classification.json" -b 64
}

eval_nce "a" "25" "model"
eval_nce "a" "50" "epoch_6"
eval_nce "b" "100" "epoch_6"
eval_nce "a" "200" "epoch_8"
eval_nce "a" "250" "epoch_9"
eval_nce "e" "150" "epoch_8"
eval_nce "e" "300" "model"
eval_nce "e" "350" "epoch_9"
eval_nce "a" "450" "epoch_8"
eval_nce "a" "500" "epoch_9"
