# Search better configuration with RoBERTa-base
# - Limit the negative.
# - Small Temperature.

relbert_training() {
  MODE=${1}
  LOSS=${2}
  MODEL='roberta-large'
  EXPORT_FILE="relbert_output/eval/accuracy.csv"
  EPOCH=50
  GRAD=8
  LR=0.000005
  TEMP=0.05
  TEMP_MIN=0.01
  NSAMPLE=640
  CKPT="relbert_output/models/d.${LOSS}.${MODE}.${MODEL}.${LR}.${GRAD}.${TEMP}.${NSAMPLE}"
  relbert-train -m "${MODEL}" --mode "${MODE}" -l "${LOSS}" -e "${EPOCH}" -b 128 --n-sample ${NSAMPLE} --export "${CKPT}" --lr "${LR}" -g "${GRAD}" \
    --temperature-nce-constant "${TEMP}" \
    --temperature-nce-max "${TEMP}"\
    --temperature-nce-min "${TEMP_MIN}" \
    -t "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>"
  relbert-eval -c "${CKPT}/epoch*" --export-file ${EXPORT_FILE} --type "analogy" --return-validation-loss
}

relbert_training 'mask' "nce_logout"
relbert_training 'mask' "nce_rank"

#df = pd.read_csv('relbert_output/eval/accuracy.analogy5.csv', index_col=0)
#df['epoch'] = [int(i.rsplit('/', 1)[-1].replace('epoch_', '')) for i in df['model']]
#df = df[df['epoch'] < 50]
#out = []
#for data in df.data.unique():
#  df_ = df[df.data == data].sort_values(by=['accuracy/test']).tail(1)
#  out.append(df_)
#  print(data, df_.to_dict())
#  print()