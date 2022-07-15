# Search better configuration with RoBERTa-base
# - Limit the negative.
# - Small Temperature.

relbert_training() {
  MODEL=${1}
  MODE=${2}
  LOSS=${3}
  EPOCH=${4}
  GRAD=${5}
  LR=0.000005
  TEMP=0.05
  NSAMPLE=640
  relbert-train -m "${MODEL}" --mode "${MODE}" -l "${LOSS}" -e "${EPOCH}" -b 128 --n-sample ${NSAMPLE} \
    --export "relbert_output/models/d.${LOSS}.${MODE}.${MODEL}.${LR}.${GRAD}.${TEMP}.${NSAMPLE}" \
    -t "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" \
    --lr "${LR}" -g "${GRAD}" --temperature-nce-constant "${TEMP}"
  relbert-eval -c "relbert_output/models/*/epoch*" --export-file "relbert_output/eval/accuracy.analogy5.csv" --type "analogy"
}

relbert_training 'roberta-large' 'mask' "nce_logout" 100 8
#relbert_training 'roberta-large' 'mask' "nce_rank" 15 8
