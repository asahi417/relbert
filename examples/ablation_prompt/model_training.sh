# NCE LOSS

train_nce () {
  TEMPLATE_ID=${1}
  TEMPLATE=${2}
  MODEL_CKPT="relbert_output/ckpt/random_template/template-${TEMPLATE_ID}"
  # train
  relbert-train -m "roberta-base" -p -a -o "${MODEL_CKPT}" -b 32 -e 10 --loss nce -r 0.000005 -t "${TEMPLATE}"
  for E in 1 2 3 4 5 6 7 8 9
  do
    relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/epoch_${E}" -o "${MODEL_CKPT}/epoch_${E}/analogy.forward.json" -b 64
  done
  relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64
}

train_nce "1" "Today, I finally discovered the spaceship between <subj> and <obj> : <subj> is the <mask> of <obj>"
train_nce "2" "Today, I finally discovered Napoleon Bonaparte between <subj> and <obj> : <subj> is the <mask> of <obj>"
train_nce "3" "Today, I finally discovered football between <subj> and <obj> : <subj> is the <mask> of <obj>"
train_nce "4" "Today, I finally discovered Italy between <subj> and <obj> : <subj> is the <mask> of <obj>"
train_nce "5" "Today, I finally discovered Cardiff between <subj> and <obj> : <subj> is the <mask> of <obj>"
train_nce "6" "Today, I finally discovered the earth science between <subj> and <obj> : <subj> is the <mask> of <obj>"
train_nce "7" "Today, I finally discovered pizza between <subj> and <obj> : <subj> is the <mask> of <obj>"
train_nce "8" "Today, I finally discovered subway between <subj> and <obj> : <subj> is the <mask> of <obj>"
train_nce "9" "Today, I finally discovered ocean between <subj> and <obj> : <subj> is the <mask> of <obj>"
train_nce "10" "Today, I finally discovered Abraham Lincoln between <subj> and <obj> : <subj> is the <mask> of <obj>"

train_nce "length_1" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>"
train_nce "length_2" "Today, I finally discovered that the relation between <subj> and <obj> is <mask>"
train_nce "length_3" "I discovered that the relation between <subj> and <obj> is <mask>"
train_nce "length_4" "the relation between <subj> and <obj> is <mask>"
train_nce "length_5" "<subj> and <obj> is <mask>"

eval_nce() {
  MODEL_CKPT="relbert_output/ckpt/random_template/template-${1}/${2}"
  # for evaluation
  relbert-eval-analogy -d 'scan' 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.forward.json" -b 64
  relbert-eval-classification -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/classification.json" -b 64
}


eval_nce "1" "epoch_8"
eval_nce "2" "epoch_6"
eval_nce "3" "epoch_6"
eval_nce "4" "epoch_6"
eval_nce "5" "epoch_6"
eval_nce "6" "model"
eval_nce "7" "epoch_6"
eval_nce "8" "epoch_6"

eval_nce "9" "epoch_6"
eval_nce "10" "epoch_8"

eval_nce "length_1" "epoch_8"
eval_nce "length_2" "epoch_6"
eval_nce "length_3" "epoch_8"
eval_nce "length_4" "epoch_6"
eval_nce "length_5" "model"