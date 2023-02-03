relbert-train -d 'relbert/t_rex_relational_similarity' -n "filter_unified.min_entity_1_max_predicate_100"  -o "tmp" -b 39 -e 1 --loss triplet -s "0" -a

# TRIPLET LOSS
train_triplet () {
  TEMPLATE_ID=${1}
  TEMPLATE=${2}
  DATA_TYPE=${3}
  MODEL_ALIAS=${4}
  MODEL_CKPT="relbert_output/ckpt/triplet_t_rex/template-${TEMPLATE_ID}"

  # train
  relbert-train -a -d "relbert/t_rex_relational_similarity" -n "${DATA_TYPE}" -p -o "${MODEL_CKPT}" -b 64 -e 1 --loss triplet -t "${TEMPLATE}"

  # for evaluation
  relbert-eval-analogy -d 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' -s 'test' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64
  relbert-eval-analogy -d 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' -s 'test' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.reverse.json" -b 64 --reverse-pair
  relbert-eval-analogy -d 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' -s 'test'  -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.bidirection.json" -b 64 --bi-direction-pair
  relbert-eval-analogy -d 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64
  relbert-eval-analogy -d 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.reverse.json" -b 64 --reverse-pair
  relbert-eval-analogy -d 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.bidirection.json" -b 64 --bi-direction-pair
  relbert-eval-classification -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/classification.json" -b 64
  relbert-eval-mapping -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/relation_mapping.json" -b 64

  # upload
  relbert-push-to-hub -m "${MODEL_CKPT}/model" -a "relbert-roberta-large-triplet-${TEMPLATE_ID}-${MODEL_ALIAS}"
}

train_triplet "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" "filter_unified.min_entity_4_max_predicate_10" "t-rex-4-10"

train_triplet "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>" "filter_unified.min_entity_4_max_predicate_10" "t-rex-4-10"
train_triplet "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>" "filter_unified.min_entity_4_max_predicate_10" "t-rex-4-10"
train_triplet "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>" "filter_unified.min_entity_4_max_predicate_10" "t-rex-4-10"
train_triplet "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>" "filter_unified.min_entity_4_max_predicate_10" "t-rex-4-10"
