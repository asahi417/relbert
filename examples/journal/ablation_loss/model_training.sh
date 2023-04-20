evaluate () {
  MODEL_CKPT=${1}
  relbert-eval-analogy -d 'scan' 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.forward.json" -b 64
  relbert-eval-classification -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/classification.json" -b 64
  relbert-eval-mapping -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/relation_mapping.json" -b 64
  # upload
  relbert-push-to-hub -m "${MODEL_CKPT}" -a "${2}"
}

# TRIPLET LOSS
train_triplet () {
  TEMPLATE_ID=${1}
  TEMPLATE=${2}
  LM=${3}
  MODEL_CKPT="relbert_output/ckpt/triplet/template-${TEMPLATE_ID}"
  # train (batch 79 is the number of whole relation types in the semeval2012)
  relbert-train -p -o "${MODEL_CKPT}" -b 79 -e 1 --loss triplet -t "${TEMPLATE}" -m "${LM}"
  relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64
}

train_triplet "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>" "roberta-base"
train_triplet "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>" "roberta-base"
train_triplet "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>" "roberta-base"
train_triplet "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" "roberta-base"
train_triplet "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>" "roberta-base"

evaluate "relbert_output/ckpt/triplet/template-e/model" "relbert-roberta-base-triplet-semeval2012"


## InfoLOOB LOSS
train_iloob () {
  TEMPLATE_ID=${1}
  TEMPLATE=${2}
  LM=${3}
  MODEL_CKPT="relbert_output/ckpt/iloob/template-${TEMPLATE_ID}"
  # train
  relbert-train -p -a -o "${MODEL_CKPT}" -b 32 -e 10 --loss iloob -r 0.000005 -t "${TEMPLATE}" -m "${LM}"
  for E in 1 2 3 4 5 6 7 8 9
  do
    relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/epoch_${E}" -o "${MODEL_CKPT}/epoch_${E}/analogy.forward.json" -b 64
  done
  relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64
}

train_iloob "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>" "roberta-base"
train_iloob "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>" "roberta-base"
train_iloob "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>" "roberta-base"
train_iloob "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" "roberta-base"
train_iloob "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>" "roberta-base"

evaluate "relbert_output/ckpt/iloob/template-a/epoch_4" "relbert-roberta-base-iloob-semeval2012"
