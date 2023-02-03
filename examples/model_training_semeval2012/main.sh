
# TRIPLET LOSS
train_triplet () {
  TEMPLATE_ID=${1}
  TEMPLATE=${2}
  MODEL_CKPT="relbert_output/ckpt/triplet_semeval2012/template-${TEMPLATE_ID}"

  # train (batch 79 is the number of whole relation types in the semeval2012)
#  relbert-train -p -o "${MODEL_CKPT}" -b 79 -e 1 --loss triplet -t "${TEMPLATE}"
#  MODEL_CKPT="relbert_output/ckpt/triplet_semeval2012/template-${TEMPLATE_ID}/model"

  git clone "https://huggingface.co/relbert/relbert-roberta-large-triplet-${TEMPLATE_ID}-semeval2012"
  MODEL_CKPT="relbert-roberta-large-triplet-${TEMPLATE_ID}-semeval2012"

  # for evaluation
  relbert-eval-analogy -d 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.forward.json" -b 64
  relbert-eval-analogy -d 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.reverse.json" -b 64 --reverse-pair
  relbert-eval-analogy -d 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test'  -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.bidirection.json" -b 64 --bi-direction-pair
  relbert-eval-analogy -d 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'validation' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.forward.json" -b 64
  relbert-eval-analogy -d 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'validation' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.reverse.json" -b 64 --reverse-pair
  relbert-eval-analogy -d 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'validation' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.bidirection.json" -b 64 --bi-direction-pair
#  relbert-eval-classification -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/classification.json" -b 64
#  relbert-eval-mapping -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/relation_mapping.json" -b 64

  # upload
  relbert-push-to-hub -m "${MODEL_CKPT}" -a "relbert-roberta-large-triplet-${TEMPLATE_ID}-semeval2012"
}

train_triplet "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>"
train_triplet "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>"
train_triplet "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>"
train_triplet "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>"
train_triplet "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>"

# NCE LOSS
train_nce () {
  TEMPLATE_ID=${1}
  TEMPLATE=${2}
  MODEL_CKPT="relbert_output/ckpt/nce_semeval2012/template-${TEMPLATE_ID}"
  # train
  relbert-train -p -a -o "${MODEL_CKPT}" -b 32 -e 10 --loss nce -r 0.000005 -t "${TEMPLATE}"
  for E in 1 2 3 4 5 6 7 8 9
  do
    relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/epoch_${E}" -o "${MODEL_CKPT}/epoch_${E}/analogy.forward.json" -b 64
  done
  relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64
}

train_nce "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>"
train_nce "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>"
train_nce "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>"
train_nce "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>"
train_nce "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>"

eval_nce() {
  TEMPLATE_ID=${1}
  MODEL_CKPT="relbert_output/ckpt/nce_semeval2012/template-${TEMPLATE_ID}/${2}"
  MODEL_ALIAS=${3}

  # for evaluation
  relbert-eval-analogy -d 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.forward.json" -b 64
  relbert-eval-analogy -d 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.reverse.json" -b 64 --reverse-pair
  relbert-eval-analogy -d 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test'  -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.bidirection.json" -b 64 --bi-direction-pair
  relbert-eval-analogy -d 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'validation' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.forward.json" -b 64
  relbert-eval-analogy -d 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'validation' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.reverse.json" -b 64 --reverse-pair
  relbert-eval-analogy -d 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'validation' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.bidirection.json" -b 64 --bi-direction-pair
#  relbert-eval-classification -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/classification.json" -b 64
#  relbert-eval-mapping -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/relation_mapping.json" -b 64

  # upload
  relbert-push-to-hub -m "${MODEL_CKPT}" -a "${MODEL_ALIAS}"
}

eval_nce "a" "epoch_8" "relbert-roberta-large-nce-a-semeval2012"
eval_nce "b" "epoch_9" "relbert-roberta-large-nce-b-semeval2012"
eval_nce "c" "epoch_6" "relbert-roberta-large-nce-c-semeval2012"
eval_nce "d" "epoch_9" "relbert-roberta-large-nce-d-semeval2012"
eval_nce "e" "epoch_9" "relbert-roberta-large-nce-e-semeval2012"

## InfoLOOB LOSS
train_iloob () {
  TEMPLATE_ID=${1}
  TEMPLATE=${2}
  MODEL_CKPT="relbert_output/ckpt/iloob_semeval2012/template-${TEMPLATE_ID}"
  # train
  relbert-train -p -a -o "${MODEL_CKPT}" -b 32 -e 9 --loss iloob -r 0.000005 -t "${TEMPLATE}"
  for E in 1 3 6
  do
    relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/epoch_${E}" -o "${MODEL_CKPT}/epoch_${E}/analogy.forward.json" -b 64
  done
  relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64
}

train_iloob "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>"
train_iloob "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>"
train_iloob "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>"
train_iloob "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>"
train_iloob "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>"

eval_iloob() {
  TEMPLATE_ID=${1}
  MODEL_CKPT="relbert_output/ckpt/iloob_semeval2012/template-${TEMPLATE_ID}/${2}"
  MODEL_ALIAS=${3}

  # for evaluation
  relbert-eval-analogy -d 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.forward.json" -b 64
  relbert-eval-analogy -d 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.reverse.json" -b 64 --reverse-pair
  relbert-eval-analogy -d 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test'  -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.bidirection.json" -b 64 --bi-direction-pair
  relbert-eval-analogy -d 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'validation' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.forward.json" -b 64
  relbert-eval-analogy -d 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'validation' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.reverse.json" -b 64 --reverse-pair
  relbert-eval-analogy -d 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'validation' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.bidirection.json" -b 64 --bi-direction-pair
  relbert-eval-classification -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/classification.json" -b 64
  relbert-eval-mapping -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/relation_mapping.json" -b 64

  # upload
  relbert-push-to-hub -m "${MODEL_CKPT}" -a "${MODEL_ALIAS}"
}

#eval_nce "d" "epoch_9" "relbert-roberta-large-iloob-d-semeval2012"
#eval_nce "e" "model" "relbert-roberta-large-iloob-e-semeval2012"
