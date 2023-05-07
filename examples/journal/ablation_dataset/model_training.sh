# NCE LOSS
train_nce () {
  TEMPLATE_ID=${1}
  TEMPLATE=${2}
  DATASET=${3}
  BATCH=${4}
  MODEL_CKPT="relbert_output/ckpt/nce_${DATASET}/template-${TEMPLATE_ID}"
  # train
  relbert-train -m "roberta-base" -p -a -o "${MODEL_CKPT}" -b "${BATCH}" -e 5 --loss nce -r 0.000005 -t "${TEMPLATE}" -d "relbert/${DATASET}" --num-positive 30 --num-negative 300
  for E in 1 2 3 4
  do
    relbert-eval-analogy -d "${DATASET}" -s 'validation' -m "${MODEL_CKPT}/epoch_${E}" -o "${MODEL_CKPT}/epoch_${E}/analogy.forward.json" -b 64
  done
  relbert-eval-analogy -d "${DATASET}" -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64
}

train_nce "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>" 'conceptnet_relational_similarity' 16
train_nce "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>" 'conceptnet_relational_similarity' 16
train_nce "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>" 'conceptnet_relational_similarity' 16
train_nce "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" 'conceptnet_relational_similarity' 16
train_nce "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>" 'conceptnet_relational_similarity' 16

train_nce "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>" 'nell_relational_similarity' 16
train_nce "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>" 'nell_relational_similarity' 16
train_nce "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>" 'nell_relational_similarity' 16
train_nce "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" 'nell_relational_similarity' 16
train_nce "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>" 'nell_relational_similarity' 16

train_nce "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>" 't_rex_relational_similarity' 32
train_nce "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>" 't_rex_relational_similarity' 32
train_nce "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>" 't_rex_relational_similarity' 32
train_nce "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" 't_rex_relational_similarity' 32
train_nce "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>" 't_rex_relational_similarity' 32

eval_nce () {
  TEMPLATE_ID=${1}
  DATASET=${2}
  MODEL_CKPT="relbert_output/ckpt/nce_${DATASET}/template-${TEMPLATE_ID}/${4}"
  # for evaluation
  relbert-eval-analogy -d 'scan' 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.forward.json" -b 64
  relbert-eval-analogy -d 'scan' 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.reverse.json" -b 64 --reverse-pair
  relbert-eval-analogy -d 'scan' 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test'  -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.bidirection.json" -b 64 --bi-direction-pair
  relbert-eval-analogy -d 'scan' 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'validation' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.forward.json" -b 64
  relbert-eval-analogy -d 'scan' 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'validation' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.reverse.json" -b 64 --reverse-pair
  relbert-eval-analogy -d 'scan' 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'validation' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.bidirection.json" -b 64 --bi-direction-pair
  relbert-eval-classification -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/classification.json" -b 64
  relbert-eval-mapping -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/relation_mapping.json" -b 64
  relbert-push-to-hub -m "${MODEL_CKPT}" -a "${3}"
}

eval_nce "d" 'conceptnet_relational_similarity' "relbert-roberta-base-nce-conceptnet" "model"
eval_nce "e" 'nell_relational_similarity' "relbert-roberta-base-nce-nell" "model"
eval_nce "a" 't_rex_relational_similarity' "relbert-roberta-base-nce-t-rex" "epoch_4"
