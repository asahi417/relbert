# NCE LOSS
train_nce () {
  TEMPLATE_ID=${1}
  TEMPLATE=${2}
  NAME=${3}
  MODEL_CKPT="relbert_output/ckpt/nce_combined.${NAME}/template-${TEMPLATE_ID}"
  # train
  relbert-train -p -o "${MODEL_CKPT}" -b 64 -e 20 --loss nce -r 0.000005 -t "${TEMPLATE}" -d 'relbert/relational_similarity' -n "${NAME}" --num-negative 300
  for E in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
  do
    relbert-eval-analogy -d 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test' -m "${MODEL_CKPT}/epoch_${E}" -o "${MODEL_CKPT}/epoch_${E}/analogy.forward.json" -b 64
    relbert-eval-analogy -d 'nell_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/epoch_${E}" -o "${MODEL_CKPT}/epoch_${E}/analogy.forward.json" -b 64
  done
  relbert-eval-analogy -d 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64
  relbert-eval-analogy -d 'nell_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64
}

#for N in 'nell_relational_similarity.semeval2012_relational_similarity' 'semeval2012_relational_similarity.t_rex_relational_similarity' 'nell_relational_similarity.semeval2012_relational_similarity.t_rex_relational_similarity'
for N in 'nell_relational_similarity.semeval2012_relational_similarity' 'semeval2012_relational_similarity.t_rex_relational_similarity'
do
#  train_nce "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>" "${N}"
  train_nce "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>" "${N}"
#  train_nce "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>" "${N}"
  train_nce "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" "${N}"
#  train_nce "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>" "${N}"
done

N="nell_relational_similarity.semeval2012_relational_similarity.t_rex_relational_similarity"
train_nce "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>" "${N}"
train_nce "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" "${N}"


eval_nce() {
  TEMPLATE_ID=${1}
  MODEL_CKPT="relbert_output/ckpt/nce_nell/template-${TEMPLATE_ID}/${2}"
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

eval_nce "a" "epoch_8" "relbert-roberta-large-nce-a-nell"
eval_nce "b" "epoch_8" "relbert-roberta-large-nce-b-nell"
eval_nce "c" "epoch_8" "relbert-roberta-large-nce-c-nell"
eval_nce "d" "epoch_8" "relbert-roberta-large-nce-d-nell"
eval_nce "e" "epoch_9" "relbert-roberta-large-nce-e-nell"
