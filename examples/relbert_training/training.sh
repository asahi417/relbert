
# Search better configuration with RoBERTa-base
relbert_training() {
  MODEL=${1}
  EPOCH=${2}
  BATCH=${3}
  LOSS=${4}
  PREFIX=${5}
  PROMPT=${6}
  MODE=${7}
  for RATIO in 0.25 0.5 0.75
  do
    relbert-train -m "${MODEL}" --mode ${MODE} -l "${LOSS}" -e "${EPOCH}" -b "${BATCH}" \
      --batch-positive-ratio "${RATIO}" --export "relbert_output/models_1/${PREFIX}.${LOSS}.${MODE}.${MODEL}.${RATIO}" \
      -t "${PROMPT}"
  done
  relbert-eval -c "relbert_output/models/${PREFIX}.${LOSS}.${MODE}.${MODEL}.*/epoch*" \
    --export-file "relbert_output/eval/accuracy.analogy.csv" --type "analogy"
}

relbert_training 'roberta-large' 10 128 "nce_rank" "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>" "mask"
relbert_training 'roberta-large' 10 128 "nce_rank" "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" "mask"
relbert_training 'roberta-large' 10 128 "nce_rank" "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>" "mask"



relbert_training 'roberta-base' 10 64 "nce_rank"
relbert_training 'roberta-base' 10 64 "nce_logout"
relbert_training 'roberta-base' 10 64 "nce_login"


# RelBERT training with RoBERTa Large

