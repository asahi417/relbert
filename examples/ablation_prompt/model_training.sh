# NCE LOSS

train_nce () {
  TEMPLATE_ID=${1}
  TEMPLATE=${2}
  MODEL_CKPT="relbert_output/ckpt/random_template/template-${TEMPLATE_ID}"
  # train
  relbert-train -m "roberta-base" -p -a -o "${MODEL_CKPT}" -b 32 -e 10 --loss nce -r 0.000005 -t "${TEMPLATE}"
  for E in 1 2 3 4 5 6 7 8 9
  do
    relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/epoch_${E}" -o "${MODEL_CKPT}/epoch_${E}/analogy.forward.json" -b 64
  done
  relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64
}

train_nce "1" "Today, I finally discovered the spaceship between [h] and [t] : [h] is the <mask> of [t]"
train_nce "2" "Today, I finally discovered Napoleon Bonaparte between [h] and [t] : [h] is the <mask> of [t]"
train_nce "3" "Today, I finally discovered football between [h] and [t] : [h] is the <mask> of [t]"
train_nce "4" "Today, I finally discovered Italy between [h] and [t] : [h] is the <mask> of [t]"
train_nce "5" "Today, I finally discovered Cardiff between [h] and [t] : [h] is the <mask> of [t]"
train_nce "6" "Today, I finally discovered the earth science between [h] and [t] : [h] is the <mask> of [t]"
train_nce "7" "Today, I finally discovered pizza between [h] and [t] : [h] is the <mask> of [t]"
train_nce "8" "Today, I finally discovered subway between [h] and [t] : [h] is the <mask> of [t]"
train_nce "9" "Today, I finally discovered ocean between [h] and [t] : [h] is the <mask> of [t]"
train_nce "10" "Today, I finally discovered Abraham Lincoln between [h] and [t] : [h] is the <mask> of [t]"

eval_nce() {
  TEMPLATE_ID=${1}
  MODEL_ALIAS=${3}
  LM=${4}
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

  # upload
  relbert-push-to-hub -m "${MODEL_CKPT}" -a "${MODEL_ALIAS}"
}


eval_nce "a" "epoch_6" "relbert-bert-base-nce-a-semeval2012" "bert-base-cased"
eval_nce "b" "epoch_8" "relbert-bert-base-nce-b-semeval2012" "bert-base-cased"
eval_nce "c" "epoch_6" "relbert-bert-base-nce-c-semeval2012" "bert-base-cased"
eval_nce "d" "epoch_7" "relbert-bert-base-nce-d-semeval2012" "bert-base-cased"
eval_nce "e" "epoch_9" "relbert-bert-base-nce-e-semeval2012" "bert-base-cased"

eval_nce "a" "epoch_5" "relbert-albert-base-nce-a-semeval2012" "albert-base-v2"
eval_nce "b" "epoch_8" "relbert-albert-base-nce-b-semeval2012" "albert-base-v2"
eval_nce "c" "epoch_8" "relbert-albert-base-nce-c-semeval2012" "albert-base-v2"
eval_nce "d" "epoch_5" "relbert-albert-base-nce-d-semeval2012" "albert-base-v2"
eval_nce "e" "epoch_9" "relbert-albert-base-nce-e-semeval2012" "albert-base-v2"

eval_nce "a" "epoch_8" "relbert-roberta-base-nce-a-semeval2012" "roberta-base"
eval_nce "b" "epoch_5" "relbert-roberta-base-nce-b-semeval2012" "roberta-base"
eval_nce "c" "epoch_7" "relbert-roberta-base-nce-c-semeval2012" "roberta-base"
eval_nce "d" "epoch_6" "relbert-roberta-base-nce-d-semeval2012" "roberta-base"
eval_nce "e" "epoch_8" "relbert-roberta-base-nce-e-semeval2012" "roberta-base"
