train () {
  LM=${1}
  TEMPLATE_ID=${2}
  TEMPLATE=${3}
  NUM=${4}
  SEED=${5}
  MODEL_CKPT="relbert_output/ckpt/${LM}.${NUM}.${SEED}/template-${TEMPLATE_ID}"
  relbert-train -m "${LM}" -p -a -o "${MODEL_CKPT}" -b 32 -e 10 --loss nce -r 0.000005 -t "${TEMPLATE}" --num-negative "${NUM}" -s "${SEED}"
  for E in 1 2 3 4 5 6 7 8 9
  do
    relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/epoch_${E}" -o "${MODEL_CKPT}/epoch_${E}/analogy.forward.json" -b 64
  done
  relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64
}

for S in 0 1 2
do
  train "roberta-large" "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>" 100 "${S}"
  train "roberta-large" "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>" 100 "${S}"
  train "roberta-large" "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>" 100 "${S}"
  train "roberta-large" "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>" 100 "${S}"
  train "roberta-large" "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" 100 "${S}"

  train "roberta-base" "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>" 400 "${S}"
  train "roberta-base" "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>" 400 "${S}"
  train "roberta-base" "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>" 400 "${S}"
  train "roberta-base" "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>" 400 "${S}"
  train "roberta-base" "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" 400 "${S}"
done

S=0
train "roberta-large" "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>" 100 "${S}"
train "roberta-large" "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>" 100 "${S}"
train "roberta-large" "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>" 100 "${S}"
train "roberta-large" "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>" 100 "${S}"
train "roberta-large" "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" 100 "${S}"


evaluate () {
  LM=${1}
  TEMPLATE_ID=${2}
  NUM=${3}
  SEED=${4}
  MODEL_FILE=${5}
  MODEL_ALIAS=${6}
  MODEL_CKPT="relbert_output/ckpt/${LM}.${NUM}.${SEED}/template-${TEMPLATE_ID}/${MODEL_FILE}"
  relbert-eval-analogy -d 'scan' 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.forward.json" -b 64
  relbert-eval-classification -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/classification.json" -b 64
  relbert-eval-mapping -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/relation_mapping.json" -b 64
  # upload
  relbert-push-to-hub -m "${MODEL_CKPT}" -a "${MODEL_ALIAS}"
}

evaluate "roberta-base" "a" "400" "0" "epoch_8" "relbert-roberta-base-nce-semeval2012-0"
evaluate "roberta-base" "e" "400" "1" "model" "relbert-roberta-base-nce-semeval2012-1"
evaluate "roberta-base" "e" "400" "2" "epoch_9" "relbert-roberta-base-nce-semeval2012-2"

evaluate "roberta-large" "a" "100" "0" "epoch_8" "relbert-roberta-large-nce-semeval2012-0"
evaluate "roberta-large" "d" "100" "1" "epoch_9" "relbert-roberta-large-nce-semeval2012-1"
evaluate "roberta-large" "d" "100" "2" "epoch_8" "relbert-roberta-large-nce-semeval2012-2"

evaluate "roberta-large" "d" "400" "0" "epoch_9" "relbert-roberta-base-nce-semeval2012-0-400"