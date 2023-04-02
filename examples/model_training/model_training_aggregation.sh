# NCE LOSS
train_nce () {
  TEMPLATE_ID=${1}
  TEMPLATE=${2}
  LM=${3}
  AGGREGATION_MODE=${4}
  MODEL_CKPT="relbert_output/ckpt/nce_aggregate_${LM}_${AGGREGATION_MODE}/template-${TEMPLATE_ID}"
  # train
  relbert-train -m "${LM}" -p -a -o "${MODEL_CKPT}" -b 32 -e 10 --loss nce -r 0.000005 -t "${TEMPLATE}" --aggregation-mode "${AGGREGATION_MODE}"
  for E in 1 2 3 4 5 6 7 8 9
  do
    relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/epoch_${E}" -o "${MODEL_CKPT}/epoch_${E}/analogy.forward.json" -b 64
  done
  relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64
}

train_nce "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>" "roberta-base" "mask"
train_nce "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>" "roberta-base" "mask"
train_nce "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>" "roberta-base" "mask"
train_nce "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" "roberta-base" "mask"
train_nce "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>" "roberta-base" "mask"

train_nce "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>" "roberta-base" "average"
train_nce "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>" "roberta-base" "average"
train_nce "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>" "roberta-base" "average"
train_nce "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" "roberta-base" "average"
train_nce "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>" "roberta-base" "average"


eval_nce() {
  TEMPLATE_ID=${1}
  MODEL_ALIAS=${3}
  LM=${4}
  AGGREGATION_MODE=${5}
  MODEL_CKPT="relbert_output/ckpt/nce_aggregate_${LM}_${AGGREGATION_MODE}/template-${TEMPLATE_ID}/${2}"


  # for evaluation
  relbert-eval-analogy -d 'scan' 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.forward.json" -b 64
  relbert-eval-analogy -d 'scan' 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.reverse.json" -b 64 --reverse-pair
  relbert-eval-analogy -d 'scan' 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test'  -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.bidirection.json" -b 64 --bi-direction-pair
  relbert-eval-analogy -d 'scan' 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'validation' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.forward.json" -b 64
  relbert-eval-analogy -d 'scan' 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'validation' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.reverse.json" -b 64 --reverse-pair
  relbert-eval-analogy -d 'scan' 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'validation' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.bidirection.json" -b 64 --bi-direction-pair
  relbert-eval-classification -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/classification.json" -b 64
  relbert-eval-mapping -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/relation_mapping.json" -b 64

  # upload
  relbert-push-to-hub -m "${MODEL_CKPT}" -a "${MODEL_ALIAS}"
}

eval_nce "a" "epoch_9" "relbert-roberta-base-nce-a-semeval2012-mask" "roberta-base" "mask"
eval_nce "b" "epoch_5" "relbert-roberta-base-nce-b-semeval2012-mask" "roberta-base" "mask"
eval_nce "c" "epoch_9" "relbert-roberta-base-nce-c-semeval2012-mask" "roberta-base" "mask"
eval_nce "d" "model" "relbert-roberta-base-nce-d-semeval2012-mask" "roberta-base" "mask"
eval_nce "e" "epoch_8" "relbert-roberta-base-nce-e-semeval2012-mask" "roberta-base" "mask"
eval_nce "a" "epoch_9" "relbert-roberta-base-nce-a-semeval2012-average" "roberta-base" "average"
eval_nce "b" "epoch_9" "relbert-roberta-base-nce-b-semeval2012-average" "roberta-base" "average"
eval_nce "c" "epoch_6" "relbert-roberta-base-nce-c-semeval2012-average" "roberta-base" "average"
eval_nce "d" "epoch_7" "relbert-roberta-base-nce-d-semeval2012-average" "roberta-base" "average"
eval_nce "e" "model" "relbert-roberta-base-nce-e-semeval2012-average" "roberta-base" "average"
