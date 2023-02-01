# TRIPLET LOSS
train_triplet () {
  TEMPLATE_ID=${1}
  TEMPLATE=${2}
  MODEL_CKPT="relbert_output/ckpt/triplet_semeval2012/template-${TEMPLATE_ID}"

  # train
  relbert-train -p -o "${MODEL_CKPT}" -b 64 -e 5 --loss triplet -t "${TEMPLATE}"

  # for evaluation
  relbert-eval-analogy -d 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' -s 'test' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64
  relbert-eval-analogy -d 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' -s 'test' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.reverse.json" -b 64 --reverse-pair
  relbert-eval-analogy -d 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' -s 'test'  -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.bidirection.json" -b 64 --bi-direction-pair
  relbert-eval-analogy -d 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64
  relbert-eval-analogy -d 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.reverse.json" -b 64 --reverse-pair
  relbert-eval-analogy -d 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.bidirection.json" -b 64 --bi-direction-pair
#  relbert-eval-classification -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/classification.json" -b 64
#  relbert-eval-mapping -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/relation_mapping.json" -b 64 --overwrite
#
#  # upload
#  relbert-push-to-hub -m "${MODEL_CKPT}/model" -a "relbert-roberta-large-triplet-${TEMPLATE_ID}-semeval2012"
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

  # for evaluation
  relbert-eval-analogy -d 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' -s 'test' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64
  relbert-eval-analogy -d 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' -s 'test' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.reverse.json" -b 64 --reverse-pair
  relbert-eval-analogy -d 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' -s 'test'  -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.bidirection.json" -b 64 --bi-direction-pair
  relbert-eval-analogy -d 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64
  relbert-eval-analogy -d 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.reverse.json" -b 64 --reverse-pair
  relbert-eval-analogy -d 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.bidirection.json" -b 64 --bi-direction-pair
#  relbert-eval-classification -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/classification.json" -b 64
#  relbert-eval-mapping -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/relation_mapping.json" -b 64 --overwrite
#
#  # upload
#  relbert-push-to-hub -m "${MODEL_CKPT}/model" -a "relbert-roberta-large-nce-${TEMPLATE_ID}-semeval2012"
}

train_nce "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>"
train_nce "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>"
train_nce "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>"
train_nce "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>"
train_nce "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>"


# InfoLOOB LOSS
train_iloob () {
  TEMPLATE_ID=${1}
  TEMPLATE=${2}
  MODEL_CKPT="relbert_output/ckpt/iloob_semeval2012/template-${TEMPLATE_ID}"

  # train
  relbert-train -p -a -o "${MODEL_CKPT}" -b 32 -e 10 --loss iloob -r 0.000005 -t "${TEMPLATE}"

  # for evaluation
  relbert-eval-analogy -d 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' -s 'test' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64
  relbert-eval-analogy -d 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' -s 'test' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.reverse.json" -b 64 --reverse-pair
  relbert-eval-analogy -d 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' -s 'test'  -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.bidirection.json" -b 64 --bi-direction-pair
  relbert-eval-analogy -d 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64
  relbert-eval-analogy -d 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.reverse.json" -b 64 --reverse-pair
  relbert-eval-analogy -d 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.bidirection.json" -b 64 --bi-direction-pair
  relbert-eval-classification -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/classification.json" -b 64
  relbert-eval-mapping -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/relation_mapping.json" -b 64 --overwrite

  # upload
  relbert-push-to-hub -m "${MODEL_CKPT}/model" -a "relbert-roberta-large-iloob-${TEMPLATE_ID}-semeval2012"
}

train_iloob "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>"
train_iloob "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>"
train_iloob "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>"
train_iloob "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>"
train_iloob "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>"
