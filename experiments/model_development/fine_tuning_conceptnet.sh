relbert_training() {
  MODEL='roberta-large'
  EPOCH=100
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
  LOSS="nce_logout"
  LOSS_ALIAS="nce"
  CKPT="relbert_output/models/${TEMPLATE_ID}.${LOSS}.${MODE}.${MODEL}.${LR}.${GRAD}.${TEMP}.${NSAMPLE}"
  relbert-train -m "${MODEL}" --mode "${MODE}" -l "${LOSS}" -e "${EPOCH}" -b 128 --n-sample ${NSAMPLE} --export "${CKPT}" --lr "${LR}" -g "${GRAD}" \
    --temperature-nce-constant "${TEMP}" --temperature-nce-max "${TEMP}" --temperature-nce-min "${TEMP_MIN}" \
    -t "${TEMPLATE}" --data "${DATA}"
  relbert-eval -c "${CKPT}/best_model" --type "analogy"
  relbert-push-to-hub -o relbert -m "${CKPT}/best_model" -a "relbert-${MODEL}-${DATA_ALIAS}-${MODE_ALIAS}-prompt-${TEMPLATE_ID}-${LOSS_ALIAS}"
}

relbert_training 'average' 'average' "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" "relbert/conceptnet_high_confidence" "conceptnet-hc"
relbert_training 'average' 'average' "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" "relbert/conceptnet" "conceptnet"


relbert_lexical_classification () {
  DATA_ALIAS=${1}
  for LOSS in "triplet" "nce"
  do
    for PROMPT in "a" "b" "c" "d" "e"
    do
      for METHOD in "mask" "average" "average-no-mask"
      do
        CKPT="relbert-roberta-large-${DATA_ALIAS}-${METHOD}-prompt-${PROMPT}-${LOSS}"
        git clone "https://huggingface.co/relbert/${CKPT}"
        relbert-eval --type classification -c "${CKPT}" --export-dir "${CKPT}" -b 64
        cd "${CKPT}"
        ga . && gcmsg 'model update' && gp
        cd ../
        rm -rf "${CKPT}"
      done
    done
  done
}

relbert_lexical_classification "semeval2012"