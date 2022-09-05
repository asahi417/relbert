MODEL='roberta-large'

relbert_training() {
  EPOCH=30
  GRAD=8
  LR=0.000005
  TEMP=0.05
  TEMP_MIN=0.01
  NSAMPLE=640
  MODE=${1}
  TEMPLATE_ID=${2}
  TEMPLATE=${3}
  DATA=${4}
  DATA_ALIAS=${5}
  LOSS=${6}
  LOSS_ALIAS=${7}
  CKPT="relbert_output/models/${TEMPLATE_ID}.${LOSS}.${MODE}.${MODEL}.${LR}.${GRAD}.${TEMP}.${NSAMPLE}.classification"
  relbert-train -m "${MODEL}" --mode "${MODE}" -l "${LOSS}" -e "${EPOCH}" -b 128 --n-sample ${NSAMPLE} --export "${CKPT}" --lr "${LR}" -g "${GRAD}" \
    --temperature-nce-constant "${TEMP}" --temperature-nce-max "${TEMP}" --temperature-nce-min "${TEMP_MIN}" -c \
    -t "${TEMPLATE}" --data "${DATA}" --split "train" --data-eval "relbert/conceptnet_high_confidence" --split-eval "full"
  relbert-push-to-hub -o relbert -m "${CKPT}/best_model" -a "${MODEL}-${DATA_ALIAS}-${MODE//_/-}-prompt-${TEMPLATE_ID}-${LOSS_ALIAS}-classification"
}


relbert_evaluation () {
  DATA_ALIAS=${1}
  LOSS_ALIAS=${2}
  MODE=${3}
  for PROMPT in "a" "b" "c" "d" "e"
  do
    CKPT="${MODEL}-${DATA_ALIAS}-${MODE//_/-}-prompt-${PROMPT}-${LOSS_ALIAS}-classification"
    git clone "https://huggingface.co/relbert/${CKPT}"
    relbert-eval --type analogy -c "${CKPT}" --export-dir "${CKPT}" -b 64
    relbert-eval --type classification -c "${CKPT}" --export-dir "${CKPT}" -b 64
    relbert-eval --type relation_mapping -c "${CKPT}" --export-dir "${CKPT}" -b 2048 --aggregation 'max'
    relbert-push-to-hub -o 'relbert' -a "${CKPT}" -m "${CKPT}"
    rm -rf "${CKPT}"
  done
}


experiment () {
  DATA=${1}
  DATA_ALIAS=${2}
  LOSS=${3}
  LOSS_ALIAS=${3}
  for MODE in "mask" "average" "average_no_mask"
  do
    relbert_training ${MODE} "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}"
    relbert_training ${MODE} "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}"
    relbert_training ${MODE} "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}"
    relbert_training ${MODE} "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}"
    relbert_training ${MODE} "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}"
    relbert_evaluation "${DATA_ALIAS}" "${LOSS_ALIAS}" "${MODE}"
  done
}

experiment "relbert/semeval2012_relational_similarity" "semeval2012" "nce_logout" "nce"
#experiment "relbert/semeval2012_relational_similarity" "semeval2012" "info_loob" "loob"
