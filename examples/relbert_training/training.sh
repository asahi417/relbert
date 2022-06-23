
# Search better configuration with RoBERTa-base
relbert_training() {
  MODEL=${1}
  MODE=${2}
  LOSS="nce_logout"
  for LR in 0.0001 0.0005 0.001
  do
    relbert-train -m "${MODEL}" --mode "${MODE}" -l "${LOSS}" -e 10 -b 128 --n-sample 700 \
      --export "relbert_output/models/d.${LOSS}.${MODE}.${MODEL}.${RATIO}.${LR}" \
      -t "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" \
      --lr ${LR}
  done
  relbert-eval -c "relbert_output/models/d.${LOSS}.${MODE}.${MODEL}.*/epoch*" --export-file "relbert_output/eval/accuracy.analogy.csv" --type "analogy"
}

relbert_training 'roberta-large' 'mask'
#relbert_training 'roberta-large' 'average_no_mask'


