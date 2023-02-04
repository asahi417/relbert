# NCE LOSS
train_nce () {
  TEMPLATE_ID=${1}
  TEMPLATE=${2}
  DATA_TYPE=${3}
  MODEL_CKPT="relbert_output/ckpt/nce_t_rex_${DATA_TYPE}/template-${TEMPLATE_ID}"
  # train
  relbert-train -p -a -o "${MODEL_CKPT}" -b 32 -e 10 --loss nce -r 0.000005 -t "${TEMPLATE}" -d "relbert/t_rex_relational_similarity" -n "${DATA_TYPE}"
  for E in 1 2 3 4 5 6 7 8 9
  do
    relbert-eval-analogy -d 't_rex_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/epoch_${E}" -o "${MODEL_CKPT}/epoch_${E}/analogy.forward.json" -b 64
  done
  relbert-eval-analogy -d 't_rex_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64
}

train_nce "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" "filter_unified.min_entity_4_max_predicate_10"

train_nce "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>" "filter_unified.min_entity_4_max_predicate_10"
train_nce "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>" "filter_unified.min_entity_4_max_predicate_10"
train_nce "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>" "filter_unified.min_entity_4_max_predicate_10"
train_nce "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>" "filter_unified.min_entity_4_max_predicate_10"


eval_nce() {
  TEMPLATE_ID=${1}
  DATA_TYPE=${2}
  MODEL_CKPT="relbert_output/ckpt/nce_t_rex_${DATA_TYPE}/template-${TEMPLATE_ID}/${3}"
  MODEL_ALIAS=${4}

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

eval_nce "d" "filter_unified.min_entity_4_max_predicate_10" "epoch_7" "relbert-roberta-large-nce-d-t-rex"

eval_nce "a" "filter_unified.min_entity_4_max_predicate_10" "epoch_8" "relbert-roberta-large-nce-a-t-rex"
eval_nce "b" "filter_unified.min_entity_4_max_predicate_10" "epoch_7" "relbert-roberta-large-nce-b-t-rex"
eval_nce "c" "filter_unified.min_entity_4_max_predicate_10" "epoch_9" "relbert-roberta-large-nce-c-t-rex"
eval_nce "e" "filter_unified.min_entity_4_max_predicate_10" "epoch_8" "relbert-roberta-large-nce-e-t-rex"