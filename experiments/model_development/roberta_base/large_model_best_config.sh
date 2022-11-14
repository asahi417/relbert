
# FINETUNING
MODE="mask"
TEMPLATE="I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>"
TEMPLATE_ID="d"
RANDOM_SEED=2

LANGUAGE_MODEL="roberta-large"
LANGUAGE_MODEL_ALIAS="roberta-large"
TEMP=0.05
TEMP_MIN=0.01
DATA="relbert/semeval2012_relational_similarity_v4"
DATA_ALIAS="semeval2012-v4"
LOSS="nce_logout"
LOSS_ALIAS="nce"
EPOCH=15
LR=0.000005
GRAD=8
NSAMPLE=640

CKPT="relbert_output/models/${DATA_ALIAS}.${TEMPLATE_ID}.${LOSS}.${MODE}.${LANGUAGE_MODEL}.${LR}.${GRAD}.${TEMP}.${NSAMPLE}.${RANDOM_SEED}"
MODEL_HF="${LANGUAGE_MODEL_ALIAS}-${DATA_ALIAS}-${MODE//_/-}-prompt-${TEMPLATE_ID}-${LOSS_ALIAS}-${RANDOM_SEED}"
relbert-train -m "${LANGUAGE_MODEL}" --mode "${MODE}" -l "${LOSS}" -e "${EPOCH}" -b 128 --n-sample "${NSAMPLE}" --export "${CKPT}" --lr "${LR}" -g "${GRAD}" \
  --temperature-nce-constant "${TEMP}" --temperature-nce-max "${TEMP}" --temperature-nce-min "${TEMP_MIN}" \
  -t "${TEMPLATE}" --data "${DATA}" --split "train" --random-seed "${RANDOM_SEED}"
relbert-push-to-hub -o relbert -m "${CKPT}/best_model" -a "${MODEL_HF}"

git clone "https://huggingface.co/relbert/${MODEL_HF}"
relbert-eval --overwrite --type analogy -c "${MODEL_HF}" --export-dir "${MODEL_HF}" -b 64
relbert-eval --overwrite --type classification -c "${MODEL_HF}" --export-dir "${MODEL_HF}" -b 64
relbert-eval --overwrite --type relation_mapping -c "${MODEL_HF}" --export-dir "${MODEL_HF}" -b 2048
relbert-push-to-hub -o 'relbert' -a "${MODEL_HF}" -m "${MODEL_HF}"
rm -rf "${MODEL_HF}"
