# TRIPLET LOSS
train_triplet () {
  RANDOM_SEED=${1}
  TEMPLATE_ID=${2}
  TEMPLATE=${3}
  MODEL_CKPT="relbert_output/ckpt/triplet/template-${TEMPLATE_ID}.random-${RANDOM_SEED}"
#  relbert-train -o "${MODEL_CKPT}" -b 39 -e 1 --loss triplet -t "${TEMPLATE}" -s "${RANDOM_SEED}"
#  relbert-eval-loss -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/loss.json" -b 64 --loss triplet
#  relbert-eval-analogy -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.json" -b 64
  relbert-eval-classification -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/classification.json" -b 64
#  relbert-eval-mapping -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/relation_mapping.json" -b 64
  relbert-eval-analogy-relation-data -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy_relation_dataset.json" -b 64
}

train_triplet "0" "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>"
train_triplet "0" "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>"
train_triplet "0" "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>"
train_triplet "0" "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>"
train_triplet "0" "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>"

# NCE LOSS
train_nce () {
  RANDOM_SEED=${1}
  TEMPLATE_ID=${2}
  TEMPLATE=${3}
  MODEL_CKPT="relbert_output/ckpt/nce/template-${TEMPLATE_ID}.random-${RANDOM_SEED}"
#  relbert-train -o "${MODEL_CKPT}" -b 32 -e 10 --loss nce -r 0.000005 -t "${TEMPLATE}" -s "${RANDOM_SEED}"
#  relbert-eval-loss -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/loss.json" -b 32 --loss nce
#  relbert-eval-analogy -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.json" -b 32
#  relbert-eval-mapping -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/relation_mapping.json" -b 64
  relbert-eval-classification -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/classification.json" -b 32
}

train_nce "0" "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>"
train_nce "0" "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>"
train_nce "0" "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>"
train_nce "0" "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>"
train_nce "0" "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>"


# InfoLOOB LOSS
train_iloob () {
  RANDOM_SEED=${1}
  TEMPLATE_ID=${2}
  TEMPLATE=${3}
  MODEL_CKPT="relbert_output/ckpt/iloob/template-${TEMPLATE_ID}.random-${RANDOM_SEED}"
  relbert-train -o "${MODEL_CKPT}" -b 32 -e 10 --loss iloob -r 0.000005 -t "${TEMPLATE}" -s "${RANDOM_SEED}"
  relbert-eval-loss -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/loss.json" -b 32 --loss iloob
  relbert-eval-analogy -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.json" -b 32
  relbert-eval-mapping -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/relation_mapping.json" -b 64
  relbert-eval-classification -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/classification.json" -b 32
}

train_iloob "0" "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>"
train_iloob "0" "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>"
train_iloob "0" "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>"
train_iloob "0" "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>"
train_iloob "0" "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>"

