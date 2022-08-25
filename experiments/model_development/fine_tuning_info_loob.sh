relbert_training_info_loob() {
  MODEL='roberta-large'
  EPOCH=30
  GRAD=8
  LR=0.000005
  TEMP=0.05
  TEMP_MIN=0.01
  NSAMPLE=640
  MODE=${1}
  MODE_ALIAS=${2}
  TEMPLATE_ID=${3}
  TEMPLATE=${4}
  DATA=${5}
  DATA_ALIAS=${6}
  LOSS="info_loob"
  LOSS_ALIAS="loob"
  CKPT="relbert_output/models/${TEMPLATE_ID}.${LOSS}.${MODE}.${MODEL}.${LR}.${GRAD}.${TEMP}.${NSAMPLE}"
  relbert-train -m "${MODEL}" --mode "${MODE}" -l "${LOSS}" -e "${EPOCH}" -b 128 --n-sample ${NSAMPLE} --export "${CKPT}" --lr "${LR}" -g "${GRAD}" \
    --temperature-nce-constant "${TEMP}" --temperature-nce-max "${TEMP}" --temperature-nce-min "${TEMP_MIN}" \
    -t "${TEMPLATE}" --data "${DATA}"
  relbert-push-to-hub -o relbert -m "${CKPT}/best_model" -a "relbert-${MODEL}-${DATA_ALIAS}-${MODE_ALIAS}-prompt-${TEMPLATE_ID}-${LOSS_ALIAS}"
}

experiment () {
  DATA=${1}
  DATA_ALIAS=${2}
  relbert_training_info_loob 'mask' 'mask' "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>" "${DATA}" "${DATA_ALIAS}"
  relbert_training_info_loob 'mask' 'mask' "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>" "${DATA}" "${DATA_ALIAS}"
  relbert_training_info_loob 'mask' 'mask' "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>" "${DATA}" "${DATA_ALIAS}"
  relbert_training_info_loob 'mask' 'mask' "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" "${DATA}" "${DATA_ALIAS}"
  relbert_training_info_loob 'mask' 'mask' "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>" "${DATA}" "${DATA_ALIAS}"

  relbert_training_info_loob 'average' 'average' "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>" "${DATA}" "${DATA_ALIAS}"
  relbert_training_info_loob 'average' 'average' "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>" "${DATA}" "${DATA_ALIAS}"
  relbert_training_info_loob 'average' 'average' "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>" "${DATA}" "${DATA_ALIAS}"
  relbert_training_info_loob 'average' 'average' "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" "${DATA}" "${DATA_ALIAS}"
  relbert_training_info_loob 'average' 'average' "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>" "${DATA}" "${DATA_ALIAS}"

  relbert_training_info_loob 'average_no_mask' 'average-no-mask' "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>" "${DATA}" "${DATA_ALIAS}"
  relbert_training_info_loob 'average_no_mask' 'average-no-mask' "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>" "${DATA}" "${DATA_ALIAS}"
  relbert_training_info_loob 'average_no_mask' 'average-no-mask' "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>" "${DATA}" "${DATA_ALIAS}"
  relbert_training_info_loob 'average_no_mask' 'average-no-mask' "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" "${DATA}" "${DATA_ALIAS}"
  relbert_training_info_loob 'average_no_mask' 'average-no-mask' "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>" "${DATA}" "${DATA_ALIAS}"
}


experiment "relbert/semeval2012_relational_similarity" "semeval2012"
experiment "relbert/semeval2012_relational_similarity_v2" "semeval2012-v2"
