
# Search better configuration with RoBERTa-base
relbert_training() {
  MODEL=${1}
  MODE=${2}
  LOSS=${3}
  EPOCH=${4}
  for LR in 0.00001 0.000005 0.000001
  do
    relbert-train -m "${MODEL}" --mode "${MODE}" -l "${LOSS}" -e "${EPOCH}" -b 128 --n-sample 640 \
      --export "relbert_output/models/d.${LOSS}.${MODE}.${MODEL}.${RATIO}.${LR}" \
      -t "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" \
      --lr ${LR}
  done
  relbert-eval -c "relbert_output/models/d.${LOSS}.${MODE}.${MODEL}.*/epoch*" --export-file "relbert_output/eval/accuracy.analogy.csv" --type "analogy"
}

relbert_training 'roberta-large' 'mask' "nce_logout" 30
#relbert_training 'roberta-large' 'average_no_mask'


