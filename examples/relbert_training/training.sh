relbert_training() {
  MODEL=${1}
  RATIO=${2}
  EPOCH=5
  BATCH=64
  for LOSS in "nce_rank" "nce_logout" "nce_login"
  do
    for MODE in "mask" "average" "average_no_mask"
      do
        relbert-train -m "${MODEL}" --mode ${MODE} -l "${LOSS}" -e "${EPOCH}" -b "${BATCH}" \
          --batch-positive-ratio "${RATIO}" --export "relbert_output/models/${LOSS}.${MODE}.${MODEL}.${RATIO}"
      done
  done
}

relbert_training 'roberta-base' 0.3

