export CKPT_DIR='./relbert_output/ckpt'
export PROMPT_DIR='./relbert_output/prompt_files'

###########
# RoBERTa #
###########
# Train with custom template
relbert-train -m roberta-large -n -p -s -t a --export ${CKPT_DIR}/roberta_custom_a
relbert-train -m roberta-large -n -p -s -t b --export ${CKPT_DIR}/roberta_custom_b
relbert-train -m roberta-large -n -p -s -t c --export ${CKPT_DIR}/roberta_custom_c
relbert-train -m roberta-large -n -p -s -t d --export ${CKPT_DIR}/roberta_custom_d
relbert-train -m roberta-large -n -p -s -t e --export ${CKPT_DIR}/roberta_custom_e
# Train with discrete tuned prompt
relbert-train -m roberta-large -n -p -s -t ${PROMPT_DIR}/roberta_d822/prompt.json --export ${CKPT_DIR}/roberta_auto_d822
relbert-train -m roberta-large -n -p -s -t ${PROMPT_DIR}/roberta_d923/prompt.json --export ${CKPT_DIR}/roberta_auto_d923
relbert-train -m roberta-large -n -p -s -t ${PROMPT_DIR}/roberta_d833/prompt.json --export ${CKPT_DIR}/roberta_auto_d833
relbert-train -m roberta-large -n -p -s -t ${PROMPT_DIR}/roberta_d823/prompt.json --export ${CKPT_DIR}/roberta_auto_d823
relbert-train -m roberta-large -n -p -s -t ${PROMPT_DIR}/roberta_d832/prompt.json --export ${CKPT_DIR}/roberta_auto_d832
relbert-train -m roberta-large -n -p -s -t ${PROMPT_DIR}/roberta_d922/prompt.json --export ${CKPT_DIR}/roberta_auto_d922
relbert-train -m roberta-large -n -p -s -t ${PROMPT_DIR}/roberta_d932/prompt.json --export ${CKPT_DIR}/roberta_auto_d932
relbert-train -m roberta-large -n -p -s -t ${PROMPT_DIR}/roberta_d933/prompt.json --export ${CKPT_DIR}/roberta_auto_d933
# Train with continuous tuned prompt
relbert-train -m roberta-large -n -p -s -t ${PROMPT_DIR}/roberta_c822/prompt.json --export ${CKPT_DIR}/roberta_auto_c822
relbert-train -m roberta-large -n -p -s -t ${PROMPT_DIR}/roberta_c923/prompt.json --export ${CKPT_DIR}/roberta_auto_c923
relbert-train -m roberta-large -n -p -s -t ${PROMPT_DIR}/roberta_c833/prompt.json --export ${CKPT_DIR}/roberta_auto_c833
relbert-train -m roberta-large -n -p -s -t ${PROMPT_DIR}/roberta_c823/prompt.json --export ${CKPT_DIR}/roberta_auto_c823
relbert-train -m roberta-large -n -p -s -t ${PROMPT_DIR}/roberta_c832/prompt.json --export ${CKPT_DIR}/roberta_auto_c832
relbert-train -m roberta-large -n -p -s -t ${PROMPT_DIR}/roberta_c922/prompt.json --export ${CKPT_DIR}/roberta_auto_c922
relbert-train -m roberta-large -n -p -s -t ${PROMPT_DIR}/roberta_c932/prompt.json --export ${CKPT_DIR}/roberta_auto_c932
relbert-train -m roberta-large -n -p -s -t ${PROMPT_DIR}/roberta_c933/prompt.json --export ${CKPT_DIR}/roberta_auto_c933

##########
# ALBERT #
##########
# Train with custom template
relbert-train -m albert-xlarge-v1 -n -p -s -t a --export ${CKPT_DIR}/albert_custom_a
relbert-train -m albert-xlarge-v1 -n -p -s -t b --export ${CKPT_DIR}/albert_custom_b
relbert-train -m albert-xlarge-v1 -n -p -s -t c --export ${CKPT_DIR}/albert_custom_c
relbert-train -m albert-xlarge-v1 -n -p -s -t d --export ${CKPT_DIR}/albert_custom_d
relbert-train -m albert-xlarge-v1 -n -p -s -t e --export ${CKPT_DIR}/albert_custom_e
# Train with discrete tuned prompt
relbert-train -m albert-xlarge-v1 -n -p -s -t ${PROMPT_DIR}/albert_d822/prompt.json --export ${CKPT_DIR}/albert_auto_d822
relbert-train -m albert-xlarge-v1 -n -p -s -t ${PROMPT_DIR}/albert_d923/prompt.json --export ${CKPT_DIR}/albert_auto_d923
relbert-train -m albert-xlarge-v1 -n -p -s -t ${PROMPT_DIR}/albert_d833/prompt.json --export ${CKPT_DIR}/albert_auto_d833
relbert-train -m albert-xlarge-v1 -n -p -s -t ${PROMPT_DIR}/albert_d823/prompt.json --export ${CKPT_DIR}/albert_auto_d823
relbert-train -m albert-xlarge-v1 -n -p -s -t ${PROMPT_DIR}/albert_d832/prompt.json --export ${CKPT_DIR}/albert_auto_d832
relbert-train -m albert-xlarge-v1 -n -p -s -t ${PROMPT_DIR}/albert_d922/prompt.json --export ${CKPT_DIR}/albert_auto_d922
relbert-train -m albert-xlarge-v1 -n -p -s -t ${PROMPT_DIR}/albert_d932/prompt.json --export ${CKPT_DIR}/albert_auto_d932
relbert-train -m albert-xlarge-v1 -n -p -s -t ${PROMPT_DIR}/albert_d933/prompt.json --export ${CKPT_DIR}/albert_auto_d933
# Train with continuous tuned prompt
relbert-train -m albert-xlarge-v1 -n -p -s -t ${PROMPT_DIR}/albert_c822/prompt.json --export ${CKPT_DIR}/albert_auto_c822
relbert-train -m albert-xlarge-v1 -n -p -s -t ${PROMPT_DIR}/albert_c923/prompt.json --export ${CKPT_DIR}/albert_auto_c923
relbert-train -m albert-xlarge-v1 -n -p -s -t ${PROMPT_DIR}/albert_c833/prompt.json --export ${CKPT_DIR}/albert_auto_c833
relbert-train -m albert-xlarge-v1 -n -p -s -t ${PROMPT_DIR}/albert_c823/prompt.json --export ${CKPT_DIR}/albert_auto_c823
relbert-train -m albert-xlarge-v1 -n -p -s -t ${PROMPT_DIR}/albert_c832/prompt.json --export ${CKPT_DIR}/albert_auto_c832
relbert-train -m albert-xlarge-v1 -n -p -s -t ${PROMPT_DIR}/albert_c922/prompt.json --export ${CKPT_DIR}/albert_auto_c922
relbert-train -m albert-xlarge-v1 -n -p -s -t ${PROMPT_DIR}/albert_c932/prompt.json --export ${CKPT_DIR}/albert_auto_c932
relbert-train -m albert-xlarge-v1 -n -p -s -t ${PROMPT_DIR}/albert_c933/prompt.json --export ${CKPT_DIR}/albert_auto_c933


########
# BERT #
########
# Train with custom template
relbert-train -m bert-large-cased -n -p -s -t a --export ${CKPT_DIR}/bert_custom_a
relbert-train -m bert-large-cased -n -p -s -t b --export ${CKPT_DIR}/bert_custom_b
relbert-train -m bert-large-cased -n -p -s -t c --export ${CKPT_DIR}/bert_custom_c
relbert-train -m bert-large-cased -n -p -s -t d --export ${CKPT_DIR}/bert_custom_d
relbert-train -m bert-large-cased -n -p -s -t e --export ${CKPT_DIR}/bert_custom_e

# Train with discrete tuned prompt
relbert-train -m bert-large-cased -n -p -s -t ${PROMPT_DIR}/bert_d822/prompt.json --export ${CKPT_DIR}/bert_auto_d822
relbert-train -m bert-large-cased -n -p -s -t ${PROMPT_DIR}/bert_d923/prompt.json --export ${CKPT_DIR}/bert_auto_d923
relbert-train -m bert-large-cased -n -p -s -t ${PROMPT_DIR}/bert_d833/prompt.json --export ${CKPT_DIR}/bert_auto_d833
relbert-train -m bert-large-cased -n -p -s -t ${PROMPT_DIR}/bert_d823/prompt.json --export ${CKPT_DIR}/bert_auto_d823
relbert-train -m bert-large-cased -n -p -s -t ${PROMPT_DIR}/bert_d832/prompt.json --export ${CKPT_DIR}/bert_auto_d832
relbert-train -m bert-large-cased -n -p -s -t ${PROMPT_DIR}/bert_d922/prompt.json --export ${CKPT_DIR}/bert_auto_d922
relbert-train -m bert-large-cased -n -p -s -t ${PROMPT_DIR}/bert_d932/prompt.json --export ${CKPT_DIR}/bert_auto_d932
relbert-train -m bert-large-cased -n -p -s -t ${PROMPT_DIR}/bert_d933/prompt.json --export ${CKPT_DIR}/bert_auto_d933

# Train with continuous tuned prompt
relbert-train -m bert-large-cased -n -p -s -t ${PROMPT_DIR}/bert_c822/prompt.json --export ${CKPT_DIR}/bert_auto_c822
relbert-train -m bert-large-cased -n -p -s -t ${PROMPT_DIR}/bert_c923/prompt.json --export ${CKPT_DIR}/bert_auto_c923
relbert-train -m bert-large-cased -n -p -s -t ${PROMPT_DIR}/bert_c833/prompt.json --export ${CKPT_DIR}/bert_auto_c833
relbert-train -m bert-large-cased -n -p -s -t ${PROMPT_DIR}/bert_c823/prompt.json --export ${CKPT_DIR}/bert_auto_c823
relbert-train -m bert-large-cased -n -p -s -t ${PROMPT_DIR}/bert_c832/prompt.json --export ${CKPT_DIR}/bert_auto_c832
relbert-train -m bert-large-cased -n -p -s -t ${PROMPT_DIR}/bert_c922/prompt.json --export ${CKPT_DIR}/bert_auto_c922
relbert-train -m bert-large-cased -n -p -s -t ${PROMPT_DIR}/bert_c932/prompt.json --export ${CKPT_DIR}/bert_auto_c932
relbert-train -m bert-large-cased -n -p -s -t ${PROMPT_DIR}/bert_c933/prompt.json --export ${CKPT_DIR}/bert_auto_c933
