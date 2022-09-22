
# FINETUNING FUNCTION
TEMP=0.05
TEMP_MIN=0.01


finetuning() {
  MODE=${1}
  TEMPLATE_ID=${2}
  TEMPLATE=${3}
  DATA=${4}
  DATA_ALIAS=${5}
  LOSS=${6}
  LOSS_ALIAS=${7}
  EPOCH=${8}
  LR=${9}
  GRAD=${10}
  NSAMPLE=${11}
  CKPT="relbert_output/models/${TEMPLATE_ID}.${LOSS}.${MODE}.roberta-large.${LR}.${GRAD}.${TEMP}.${NSAMPLE}"
  MODEL_HF="roberta-large-${DATA_ALIAS}-${MODE//_/-}-prompt-${TEMPLATE_ID}-${LOSS_ALIAS}"
  relbert-train -m "roberta-large" --mode "${MODE}" -l "${LOSS}" -e "${EPOCH}" -b 128 --n-sample "${NSAMPLE}" --export "${CKPT}" --lr "${LR}" -g "${GRAD}" \
    --temperature-nce-constant "${TEMP}" --temperature-nce-max "${TEMP}" --temperature-nce-min "${TEMP_MIN}" \
    -t "${TEMPLATE}" --data "${DATA}" --split "train"
  relbert-push-to-hub -o relbert -m "${CKPT}/best_model" -a "${MODEL_HF}"

  git clone "https://huggingface.co/relbert/${MODEL_HF}"
  relbert-eval --overwrite --type analogy -c "${MODEL_HF}" --export-dir "${MODEL_HF}" -b 64
  relbert-eval --overwrite --type classification -c "${MODEL_HF}" --export-dir "${MODEL_HF}" -b 64
  relbert-eval --overwrite --type relation_mapping -c "${MODEL_HF}" --export-dir "${MODEL_HF}" -b 2048
  relbert-push-to-hub -o 'relbert' -a "${MODEL_HF}" -m "${MODEL_HF}"
  rm -rf "${MODEL_HF}"
}

finetuning_classification() {
  MODE=${1}
  TEMPLATE_ID=${2}
  TEMPLATE=${3}
  DATA=${4}
  DATA_ALIAS=${5}
  LOSS=${6}
  LOSS_ALIAS=${7}
  EPOCH=${8}
  LR=${9}
  GRAD=${10}
  NSAMPLE=${11}
  CKPT="relbert_output/models/${TEMPLATE_ID}.${LOSS}.${MODE}.roberta-large.${LR}.${GRAD}.${TEMP}.${NSAMPLE}.classification"
  MODEL_HF="roberta-large-${DATA_ALIAS}-${MODE//_/-}-prompt-${TEMPLATE_ID}-${LOSS_ALIAS}-classification"
  relbert-train -m "roberta-large" --mode "${MODE}" -l "${LOSS}" -e "${EPOCH}" -b 128 --n-sample "${NSAMPLE}" --export "${CKPT}" --lr "${LR}" -g "${GRAD}" \
    --temperature-nce-constant "${TEMP}" --temperature-nce-max "${TEMP}" --temperature-nce-min "${TEMP_MIN}" -c \
    -t "${TEMPLATE}" --data "${DATA}" --split "train"
  relbert-push-to-hub -o relbert -m "${CKPT}/best_model" -a "${MODEL_HF}"

  git clone "https://huggingface.co/relbert/${MODEL_HF}"
  relbert-eval --overwrite --type analogy -c "${MODEL_HF}" --export-dir "${MODEL_HF}" -b 64
  relbert-eval --overwrite --type classification -c "${MODEL_HF}" --export-dir "${MODEL_HF}" -b 64
  relbert-eval --overwrite --type relation_mapping -c "${MODEL_HF}" --export-dir "${MODEL_HF}" -b 2048
  relbert-push-to-hub -o 'relbert' -a "${MODEL_HF}" -m "${MODEL_HF}"
  rm -rf "${MODEL_HF}"
}

finetuning_conceptnet_validated() {
  MODE=${1}
  TEMPLATE_ID=${2}
  TEMPLATE=${3}
  DATA=${4}
  DATA_ALIAS=${5}
  LOSS=${6}
  LOSS_ALIAS=${7}
  EPOCH=${8}
  LR=${9}
  GRAD=${10}
  NSAMPLE=${11}
  CKPT="relbert_output/models/${TEMPLATE_ID}.${LOSS}.${MODE}.roberta-large.${LR}.${GRAD}.${TEMP}.${NSAMPLE}.coneptnet-validated"
  MODEL_HF="roberta-large-${DATA_ALIAS}-${MODE//_/-}-prompt-${TEMPLATE_ID}-${LOSS_ALIAS}-conceptnet-validated"
  relbert-train -m "roberta-large" --mode "${MODE}" -l "${LOSS}" -e "${EPOCH}" -b 128 --n-sample "${NSAMPLE}" --export "${CKPT}" --lr "${LR}" -g "${GRAD}" \
    --temperature-nce-constant "${TEMP}" --temperature-nce-max "${TEMP}" --temperature-nce-min "${TEMP_MIN}" \
    -t "${TEMPLATE}" --data "${DATA}" --split "train" --data-eval "relbert/conceptnet_high_confidence" --split-eval "full"
  relbert-push-to-hub -o relbert -m "${CKPT}/best_model" -a "${MODEL_HF}"

  git clone "https://huggingface.co/relbert/${MODEL_HF}"
  relbert-eval --overwrite --type analogy -c "${MODEL_HF}" --export-dir "${MODEL_HF}" -b 64
  relbert-eval --overwrite --type classification -c "${MODEL_HF}" --export-dir "${MODEL_HF}" -b 64
  relbert-eval --overwrite --type relation_mapping -c "${MODEL_HF}" --export-dir "${MODEL_HF}" -b 2048
  relbert-push-to-hub -o 'relbert' -a "${MODEL_HF}" -m "${MODEL_HF}"
  rm -rf "${MODEL_HF}"
}

finetuning_conceptnet_validated_classification() {
  MODE=${1}
  TEMPLATE_ID=${2}
  TEMPLATE=${3}
  DATA=${4}
  DATA_ALIAS=${5}
  LOSS=${6}
  LOSS_ALIAS=${7}
  EPOCH=${8}
  LR=${9}
  GRAD=${10}
  NSAMPLE=${11}
  CKPT="relbert_output/models/${TEMPLATE_ID}.${LOSS}.${MODE}.roberta-large.${LR}.${GRAD}.${TEMP}.${NSAMPLE}.classification.conceptnet-validated"
  MODEL_HF="roberta-large-${DATA_ALIAS}-${MODE//_/-}-prompt-${TEMPLATE_ID}-${LOSS_ALIAS}-classification-conceptnet-validated"
  relbert-train -m "roberta-large" --mode "${MODE}" -l "${LOSS}" -e "${EPOCH}" -b 128 --n-sample "${NSAMPLE}" --export "${CKPT}" --lr "${LR}" -g "${GRAD}" \
    --temperature-nce-constant "${TEMP}" --temperature-nce-max "${TEMP}" --temperature-nce-min "${TEMP_MIN}" -c \
    -t "${TEMPLATE}" --data "${DATA}" --split "train" --data-eval "relbert/conceptnet_high_confidence" --split-eval "full"
  relbert-push-to-hub -o relbert -m "${CKPT}/best_model" -a "${MODEL_HF}"

  git clone "https://huggingface.co/relbert/${MODEL_HF}"
  relbert-eval --overwrite --type analogy -c "${MODEL_HF}" --export-dir "${MODEL_HF}" -b 64
  relbert-eval --overwrite --type classification -c "${MODEL_HF}" --export-dir "${MODEL_HF}" -b 64
  relbert-eval --overwrite --type relation_mapping -c "${MODEL_HF}" --export-dir "${MODEL_HF}" -b 2048
  relbert-push-to-hub -o 'relbert' -a "${MODEL_HF}" -m "${MODEL_HF}"
  rm -rf "${MODEL_HF}"
}


experiment () {
  DATA=${1}
  DATA_ALIAS=${2}
  LOSS=${3}
  LOSS_ALIAS=${4}
  EPOCH=${5}
  LR=${6}
  GRAD=${7}
  NSAMPLE=${8}
  for MODE in "mask" "average" "average_no_mask"
  do

    finetuning "${MODE}" "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}" "${EPOCH}" "${LR}" "${GRAD}" "${NSAMPLE}"
    finetuning "${MODE}" "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}" "${EPOCH}" "${LR}" "${GRAD}" "${NSAMPLE}"
    finetuning "${MODE}" "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}" "${EPOCH}" "${LR}" "${GRAD}" "${NSAMPLE}"
    finetuning "${MODE}" "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}" "${EPOCH}" "${LR}" "${GRAD}" "${NSAMPLE}"
    finetuning "${MODE}" "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}" "${EPOCH}" "${LR}" "${GRAD}" "${NSAMPLE}"

#    finetuning_classification "${MODE}" "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}" "${EPOCH}" "${LR}" "${GRAD}" "${NSAMPLE}"
#    finetuning_classification "${MODE}" "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}" "${EPOCH}" "${LR}" "${GRAD}" "${NSAMPLE}"
#    finetuning_classification "${MODE}" "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}" "${EPOCH}" "${LR}" "${GRAD}" "${NSAMPLE}"
#    finetuning_classification "${MODE}" "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}" "${EPOCH}" "${LR}" "${GRAD}" "${NSAMPLE}"
#    finetuning_classification "${MODE}" "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}" "${EPOCH}" "${LR}" "${GRAD}" "${NSAMPLE}"

#    finetuning_conceptnet_validated "${MODE}" "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}" "${EPOCH}" "${LR}" "${GRAD}" "${NSAMPLE}"
#    finetuning_conceptnet_validated "${MODE}" "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}" "${EPOCH}" "${LR}" "${GRAD}" "${NSAMPLE}"
#    finetuning_conceptnet_validated "${MODE}" "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}" "${EPOCH}" "${LR}" "${GRAD}" "${NSAMPLE}"
#    finetuning_conceptnet_validated "${MODE}" "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}" "${EPOCH}" "${LR}" "${GRAD}" "${NSAMPLE}"
#    finetuning_conceptnet_validated "${MODE}" "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}" "${EPOCH}" "${LR}" "${GRAD}" "${NSAMPLE}"
#
#    finetuning_conceptnet_validated_classification "${MODE}" "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}" "${EPOCH}" "${LR}" "${GRAD}" "${NSAMPLE}"
#    finetuning_conceptnet_validated_classification "${MODE}" "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}" "${EPOCH}" "${LR}" "${GRAD}" "${NSAMPLE}"
#    finetuning_conceptnet_validated_classification "${MODE}" "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}" "${EPOCH}" "${LR}" "${GRAD}" "${NSAMPLE}"
#    finetuning_conceptnet_validated_classification "${MODE}" "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}" "${EPOCH}" "${LR}" "${GRAD}" "${NSAMPLE}"
#    finetuning_conceptnet_validated_classification "${MODE}" "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>" "${DATA}" "${DATA_ALIAS}" "${LOSS}" "${LOSS_ALIAS}" "${EPOCH}" "${LR}" "${GRAD}" "${NSAMPLE}"

  done
}

#experiment "relbert/semeval2012_relational_similarity" "semeval2012" "info_loob" "loob" 30 0.000005 8 640
#experiment "relbert/semeval2012_relational_similarity" "semeval2012" "nce_logout" "nce" 30 0.000005 8 640
experiment "relbert/semeval2012_relational_similarity" "semeval2012" "triplet" "tri" 10 0.00005 8 320 # 0.00001 1 320
#experiment "relbert/conceptnet_high_confidence" "conceptnet" "nce_logout" "nce" 30 0.000005 8 640

