# NCE LOSS
train_nce () {
  TEMPLATE_ID=${1}
  TEMPLATE=${2}
  LM=${3}
  MODEL_CKPT="relbert_output/ckpt/nce_semeval2012_${LM}/template-${TEMPLATE_ID}"
  # train
  relbert-train -m "${LM}" -p -a -o "${MODEL_CKPT}" -b 32 -e 10 --loss nce -r 0.000005 -t "${TEMPLATE}"
  for E in 1 2 3 4 5 6 7 8 9
  do
    relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/epoch_${E}" -o "${MODEL_CKPT}/epoch_${E}/analogy.forward.json" -b 64
  done
  relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64
}

train_nce "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>" "bert-base-cased"
train_nce "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>" "bert-base-cased"
train_nce "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>" "bert-base-cased"
train_nce "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" "bert-base-cased"
train_nce "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>" "bert-base-cased"

train_nce "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>" "albert-base-v2"
train_nce "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>" "albert-base-v2"
train_nce "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>" "albert-base-v2"
train_nce "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" "albert-base-v2"
train_nce "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>" "albert-base-v2"

eval_nce() {
  TEMPLATE_ID=${1}
  LM=${3}
  MODEL_CKPT="relbert_output/ckpt/nce_semeval2012_${LM}/template-${TEMPLATE_ID}/${2}"


  # for evaluation
  relbert-eval-analogy -d 'scan' 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.forward.json" -b 64
  relbert-eval-analogy -d 'scan' 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.reverse.json" -b 64 --reverse-pair
  relbert-eval-analogy -d 'scan' 'sat_full' 'sat' 'u2' 'u4' 'google' 'bats' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'test'  -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.bidirection.json" -b 64 --bi-direction-pair
  relbert-eval-analogy -d 'scan' 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'validation' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.forward.json" -b 64
  relbert-eval-analogy -d 'scan' 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'validation' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.reverse.json" -b 64 --reverse-pair
  relbert-eval-analogy -d 'scan' 'sat' 'u2' 'u4' 'google' 'bats' 'semeval2012_relational_similarity' 't_rex_relational_similarity' 'conceptnet_relational_similarity' 'nell_relational_similarity' -s 'validation' -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/analogy.bidirection.json" -b 64 --bi-direction-pair
  relbert-eval-classification -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/classification.json" -b 64
  relbert-eval-mapping -m "${MODEL_CKPT}" -o "${MODEL_CKPT}/relation_mapping.json" -b 64
}

eval_nce "a" "epoch_6" "bert-base-cased"
eval_nce "b" "epoch_8" "bert-base-cased"
eval_nce "c" "epoch_6" "bert-base-cased"
eval_nce "d" "epoch_7" "bert-base-cased"
eval_nce "e" "epoch_9" "bert-base-cased"

eval_nce "a" "epoch_5" "albert-base-v2"
eval_nce "b" "epoch_8" "albert-base-v2"
eval_nce "c" "epoch_8" "albert-base-v2"
eval_nce "d" "epoch_5" "albert-base-v2"
eval_nce "e" "epoch_9" "albert-base-v2"

# upload
relbert-push-to-hub -m "relbert_output/ckpt/nce_semeval2012_albert-base-v2/template-a/epoch_5" -a "relbert-albert-base-nce-semeval2012"
relbert-push-to-hub -m "relbert_output/ckpt/nce_semeval2012_bert-base-cased/template-a/epoch_6" -a "relbert-bert-base-nce-semeval2012"
