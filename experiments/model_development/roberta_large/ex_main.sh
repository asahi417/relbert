# Search best config for RelSimV4 with different levels in validation
TEMP=0.05
TEMP_MIN=0.01
LANGUAGE_MODEL="roberta-large"
LANGUAGE_MODEL_ALIAS="roberta-large"
LR=0.000005
GRAD=8
NSAMPLE=320
EPOCH=10


finetuning() {
  MODE=${1}
  TEMPLATE_ID=${2}
  TEMPLATE=${3}
  DATA=${4}
  DATA_ALIAS=${5}
  LOSS=${6}
  LOSS_ALIAS=${7}
  RANDOM_SEED=${8}
  CKPT="relbert_output/models/${DATA_ALIAS}.${TEMPLATE_ID}.${LOSS}.${MODE}.${LANGUAGE_MODEL}.${LR}.${GRAD}.${TEMP}.${NSAMPLE}.${RANDOM_SEED}"
  MODEL_HF="relbert-${LANGUAGE_MODEL_ALIAS}-${DATA_ALIAS}-${MODE//_/-}-prompt-${TEMPLATE_ID}-${LOSS_ALIAS}-${RANDOM_SEED}"
  relbert-train -m "${LANGUAGE_MODEL}" --mode "${MODE}" -l "${LOSS}" -e "${EPOCH}" -b 128 --n-sample "${NSAMPLE}" --export "${CKPT}" --lr "${LR}" -g "${GRAD}" \
    --temperature-nce-constant "${TEMP}" --temperature-nce-max "${TEMP}" --temperature-nce-min "${TEMP_MIN}" \
    -t "${TEMPLATE}" --data "${DATA}" --split "train" --random-seed "${RANDOM_SEED}"
  relbert-push-to-hub -o 'relbert' -m "${CKPT}/best_model" -a "${MODEL_HF}"
  git clone "https://huggingface.co/relbert/${MODEL_HF}"
  relbert-eval --type analogy -c "${MODEL_HF}" --export-dir "${MODEL_HF}" -b 512
  relbert-eval --type classification -c "${MODEL_HF}" --export-dir "${MODEL_HF}" -b 512
  relbert-eval --type relation_mapping -c "${MODEL_HF}" --export-dir "${MODEL_HF}" -b 512
  relbert-push-to-hub -o 'relbert' -a "${MODEL_HF}" -m "${MODEL_HF}"
  rm -rf "${MODEL_HF}"
}


experiment () {
  DATA=${1}
  DATA_ALIAS=${2}
  LOSS=${3}
  LOSS_ALIAS=${4}
  R="0"
  MODE="mask"
#  for MODE in "mask" "average"
#  do
  finetuning "${MODE}" "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}" "${R}"
  finetuning "${MODE}" "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}" "${R}"
  finetuning "${MODE}" "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}" "${R}"
  finetuning "${MODE}" "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}" "${R}"
  finetuning "${MODE}" "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}" "${R}"
#  done
}

#experiment "relbert/semeval2012_relational_similarity_v6" "semeval2012-v6" "nce_logout" "nce"
#experiment "relbert/semeval2012_relational_similarity_v6" "semeval2012-v6" "triplet" "triplet"
experiment "relbert/semeval2012_relational_similarity_v6" "semeval2012-v6" "info_loob" "loob"
