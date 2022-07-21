relbert_training() {
  # fixed config
  MODEL='roberta-large'
  EXPORT_FILE="relbert_output/eval/accuracy.csv"
  EPOCH=30
  GRAD=8
  LR=0.000005
  TEMP=0.05
  TEMP_MIN=0.01
  NSAMPLE=640
  MODE=${1}
  LOSS=${2}
  TEMPLATE_ID=${3}
  TEMPLATE=${4}
  CKPT="relbert_output/models/${TEMPLATE_ID}.${LOSS}.${MODE}.${MODEL}.${LR}.${GRAD}.${TEMP}.${NSAMPLE}"
  relbert-train -m "${MODEL}" --mode "${MODE}" -l "${LOSS}" -e "${EPOCH}" -b 128 --n-sample ${NSAMPLE} --export "${CKPT}" --lr "${LR}" -g "${GRAD}" \
    --temperature-nce-constant "${TEMP}" --temperature-nce-max "${TEMP}" --temperature-nce-min "${TEMP_MIN}" \
    -t "${TEMPLATE}"
  relbert-eval -c "${CKPT}/best_model" --export-file ${EXPORT_FILE} --type "analogy" --return-validation-loss
}

relbert_training 'average_no_mask' "nce_logout" "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>"
relbert_training 'average_no_mask' "nce_logout" "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>"
relbert_training 'average_no_mask' "nce_logout" "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>"
relbert_training 'average_no_mask' "nce_logout" "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>"
relbert_training 'average_no_mask' "nce_logout" "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>"

relbert_training 'mask' "nce_logout" "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>"
relbert_training 'mask' "nce_logout" "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>"
relbert_training 'mask' "nce_logout" "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>"
relbert_training 'mask' "nce_logout" "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>"
relbert_training 'mask' "nce_logout" "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>"

relbert_training 'average' "nce_logout" "d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>"
relbert_training 'average' "nce_logout" "a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>"
relbert_training 'average' "nce_logout" "b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>"
relbert_training 'average' "nce_logout" "c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>"
relbert_training 'average' "nce_logout" "e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>"

# TODO: Implement pipeline to upload the best model to huggingface in terms of the validation error and remove all the others
#relbert_training 'mask' "nce_logout"
#relbert_training 'average_no_mask' "nce_logout"
#relbert_training 'mask' "nce_login"
#relbert_training 'average_no_mask' "nce_login"
#
#relbert_training 'average' "nce_logout"
#relbert_training 'mask' "nce_rank"

#relbert_training 'average_no_mask' "nce_rank"


df = pd.read_csv('relbert_output/eval/accuracy.csv', index_col=0)
df['epoch'] = [int(i.rsplit('/', 1)[-1].replace('epoch_', '')) for i in df['model']]
df['loss_function'] = [i.split('.')[1] for i in df['model']]
#df = df[df['epoch'] <= 30]
df = df[df['mode'] == 'average_no_mask']
df = df[df['loss_function'] == 'nce_logout']
for data in df.data.unique():
  df_ = df[df.data == data].sort_values(by=['validation_loss'], ascending=True)
#  df_ = df[df.data == data].sort_values(by=['accuracy/valid'], ascending=False)
  df_logout = df_[df_['loss_function'] == 'nce_logout'][['data', 'accuracy/test', 'epoch', 'validation_loss', 'mode', 'accuracy/full']]
#  df_login = df_[df_['loss_function'] == 'nce_login'][['data', 'accuracy/test', 'epoch', 'validation_loss', 'mode']]
  print(df_logout.head(3))
#  print(df_login.head(5))
  input()
