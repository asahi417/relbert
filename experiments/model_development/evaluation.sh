
relbert_evaluation () {
  DATA_ALIAS=${1}
  LOSS=${2}
  for PROMPT in "a" "b" "c" "d" "e"
  do
    for METHOD in "mask" "average" "average-no-mask"
    do
      CKPT="relbert-roberta-large-${DATA_ALIAS}-${METHOD}-prompt-${PROMPT}-${LOSS}"
      git clone "https://huggingface.co/relbert/${CKPT}"
      relbert-eval --type analogy -c "${CKPT}" --export-dir "${CKPT}" -b 64
      relbert-eval --type classification -c "${CKPT}" --export-dir "${CKPT}" -b 64
      relbert-eval --type relation_mapping -c "${CKPT}" --export-dir "${CKPT}" -b 2048 --aggregation 'max'
      relbert-push-to-hub -o 'relbert' -a "${CKPT}" -m "${CKPT}"
      rm -rf "${CKPT}"
    done
  done
}


relbert_evaluation "semeval2012" "nce"
relbert_evaluation "semeval2012" "triplet"
relbert_evaluation "conceptnet-hc" "nce"
