# NCE LOSS
train_nce () {
  TEMPLATE_ID=${1}
  TEMPLATE=${2}
  MODEL_CKPT="relbert_output/ckpt/nce_conceptnet/template-${TEMPLATE_ID}"
  # train
  relbert-train -p -a -o "${MODEL_CKPT}" -b 16 -e 10 --loss nce -r 0.000005 -t "${TEMPLATE}" -d 'relbert/conceptnet_relational_similarity' --num-negative 300
  for E in 1 2 3 4 5 6 7 8 9
  do
    relbert-eval-analogy --overwrite -d 'conceptnet_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/epoch_${E}" -o "${MODEL_CKPT}/epoch_${E}/analogy.forward.json" -b 64
  done
  relbert-eval-analogy --overwrite -d 'conceptnet_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64
}

train_nce "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>"
train_nce "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>"
train_nce "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>"
train_nce "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>"
train_nce "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>"

eval_nce() {
  TEMPLATE_ID=${1}
  MODEL_CKPT="relbert_output/ckpt/nce_conceptnet/template-${TEMPLATE_ID}/${2}"
  MODEL_ALIAS=${3}

  # for evaluation
  relbert-eval-analogy --overwrite -d 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.forward.json" -b 64
  relbert-eval-analogy --overwrite -d 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.reverse.json" -b 64 --reverse-pair
  relbert-eval-analogy --overwrite -d 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test'  -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.bidirection.json" -b 64 --bi-direction-pair
  relbert-eval-analogy --overwrite -d 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'validation' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.forward.json" -b 64
  relbert-eval-analogy --overwrite -d 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'validation' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.reverse.json" -b 64 --reverse-pair
  relbert-eval-analogy --overwrite -d 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'validation' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.bidirection.json" -b 64 --bi-direction-pair
  relbert-eval-classification --overwrite -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/classification.json" -b 64
  relbert-eval-mapping --overwrite -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/relation_mapping.json" -b 64

  # upload
  relbert-push-to-hub -m "${MODEL_CKPT}" -a "${MODEL_ALIAS}"
}

eval_nce "a" "model" "relbert-roberta-large-nce-a-conceptnet"
eval_nce "b" "epoch_9" "relbert-roberta-large-nce-b-conceptnet"
eval_nce "c" "model" "relbert-roberta-large-nce-c-conceptnet"
eval_nce "d" "model" "relbert-roberta-large-nce-d-conceptnet"
eval_nce "e" "model" "relbert-roberta-large-nce-e-conceptnet"
