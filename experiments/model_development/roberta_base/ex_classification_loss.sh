# Add classification loss (use the best config of the full model)
TEMP=0.05
TEMP_MIN=0.01
LANGUAGE_MODEL="roberta-base"
LANGUAGE_MODEL_ALIAS="roberta-base"
LR=0.000005
GRAD=8
NSAMPLE=640

DATA="relbert/semeval2012_relational_similarity_v4"
DATA_ALIAS="semeval2012-v4"

MODE="mask"
TEMPLATE_ID="d"
EPOCH=10
TEMPLATE="I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>"
RANDOM_SEED=2
LOSS="nce_logout"
LOSS_ALIAS="nce"
CKPT="relbert_output/models_classification_loss/${DATA_ALIAS}.${TEMPLATE_ID}.${LOSS}.${MODE}.${LANGUAGE_MODEL}.${LR}.${GRAD}.${TEMP}.${NSAMPLE}.${RANDOM_SEED}"
MODEL_HF="${LANGUAGE_MODEL_ALIAS}-${DATA_ALIAS}-${MODE//_/-}-prompt-${TEMPLATE_ID}-${LOSS_ALIAS}-${RANDOM_SEED}-ablation-classification"
relbert-train -m "${LANGUAGE_MODEL}" --mode "${MODE}" -l "${LOSS}" -e "${EPOCH}" -b 128 --n-sample "${NSAMPLE}" --export "${CKPT}" --lr "${LR}" -g "${GRAD}" \
  --temperature-nce-constant "${TEMP}" --temperature-nce-max "${TEMP}" --temperature-nce-min "${TEMP_MIN}" \
  -t "${TEMPLATE}" --data "${DATA}" --split "train" --random-seed "${RANDOM_SEED}" \
  --fix-epoch -c
relbert-push-to-hub -o relbert -m "${CKPT}/best_model" -a "${MODEL_HF}"

git clone "https://huggingface.co/relbert/${MODEL_HF}"
relbert-eval --overwrite --type analogy -c "${MODEL_HF}" --export-dir "${MODEL_HF}" -b 512
relbert-eval --overwrite --type classification -c "${MODEL_HF}" --export-dir "${MODEL_HF}" -b 512
relbert-eval --overwrite --type relation_mapping -c "${MODEL_HF}" --export-dir "${MODEL_HF}" -b 512
relbert-push-to-hub -o 'relbert' -a "${MODEL_HF}" -m "${MODEL_HF}"
rm -rf "${MODEL_HF}"
