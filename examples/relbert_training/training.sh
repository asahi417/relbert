
# Search better configuration with RoBERTa-base
relbert_training() {
  MODEL=${1}
  MODE=${2}
  LOSS=${3}
  EPOCH=${4}
  LR=0.000005
  relbert-train -m "${MODEL}" --mode "${MODE}" -l "${LOSS}" -e "${EPOCH}" -b 128 --n-sample 640 \
    --export "relbert_output/models/d.${LOSS}.${MODE}.${MODEL}.${LR}" \
    -t "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" \
    --lr ${LR}

  for LR in 0.00005 0.00001 0.000005
  do
    relbert-train -m "${MODEL}" --mode "${MODE}" -l "${LOSS}" -e "${EPOCH}" -b 128 --n-sample 640 \
      --export "relbert_output/models/d.${LOSS}.${MODE}.${MODEL}.${LR}" \
      -t "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" \
      --lr ${LR}
    relbert-eval -c "relbert_output/models/d.${LOSS}.${MODE}.${MODEL}.${LR}/epoch*" --export-file "relbert_output/eval/accuracy.analogy.csv" --type "analogy"
  done

}

relbert_training 'roberta-large' 'mask' "nce_logout" 30
#relbert_training 'roberta-large' 'average_no_mask'


relbert-train -m 'roberta-large' --mode 'mask' -l 'nce_logout' -e 30 -b 128 --n-sample 640 --lr 0.0001 \
  --export "relbert_output/models/test" \
  -t "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>"
