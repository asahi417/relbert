# NCE LOSS

train_nce () {
  TEMPLATE="Today, I finally discovered the spaceship between <subj> and <obj> : <subj> is the <mask> of <obj>"
  SEED=${1}
  MODEL_CKPT="relbert_output/ckpt/random_seed/${SEED}"
  # train
  relbert-train -m "roberta-base" -p -a -o "${MODEL_CKPT}" -b 32 -e 10 --loss nce -r 0.000005 -t "${TEMPLATE}" -s "${SEED}"
  for E in 1 2 3 4 5 6 7 8 9
  do
    relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/epoch_${E}" -o "${MODEL_CKPT}/epoch_${E}/analogy.forward.json" -b 64
  done
  relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64
}

train_nce "1"
train_nce "2"
train_nce "10"
train_nce "100"

train_nce "52"
train_nce "42"
train_nce "31"
train_nce "67"
train_nce "73"



eval_nce() {
  MODEL_CKPT="relbert_output/ckpt/random_seed/${1}/${2}"
  # for evaluation
  relbert-eval-analogy -d 'scan' 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.forward.json" -b 64
  relbert-eval-classification -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/classification.json" -b 64
}


eval_nce "1" "epoch_9"
eval_nce "2" "epoch_5"
eval_nce "10" "epoch_5"
eval_nce "100" "model"
eval_nce "52" "epoch_5"
eval_nce "42" "epoch_9"
eval_nce "31" "epoch_7"
eval_nce "67" "epoch_7"
eval_nce "73" "epoch_5"
