export PROMPT_DIR='./relbert_output/prompt_files'

############
# discrete #
############
relbert-prompt-discrete -m roberta-large --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 2 --export ${PROMPT_DIR}/roberta_d822
relbert-prompt-discrete -m roberta-large --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 2 --export ${PROMPT_DIR}/roberta_d832
relbert-prompt-discrete -m roberta-large --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 3 --export ${PROMPT_DIR}/roberta_d833
relbert-prompt-discrete -m roberta-large --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 3 --export ${PROMPT_DIR}/roberta_d823
relbert-prompt-discrete -m roberta-large --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 2 --export ${PROMPT_DIR}/roberta_d922
relbert-prompt-discrete -m roberta-large --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 3 --export ${PROMPT_DIR}/roberta_d923
relbert-prompt-discrete -m roberta-large --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 2 --export ${PROMPT_DIR}/roberta_d932
relbert-prompt-discrete -m roberta-large --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 3 --export ${PROMPT_DIR}/roberta_d933

relbert-prompt-discrete -m albert-xlarge-v1 --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 2 --export ${PROMPT_DIR}/albert_d822
relbert-prompt-discrete -m albert-xlarge-v1 --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 2 --export ${PROMPT_DIR}/albert_d832
relbert-prompt-discrete -m albert-xlarge-v1 --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 3 --export ${PROMPT_DIR}/albert_d833
relbert-prompt-discrete -m albert-xlarge-v1 --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 3 --export ${PROMPT_DIR}/albert_d823
relbert-prompt-discrete -m albert-xlarge-v1 --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 2 --export ${PROMPT_DIR}/albert_d922
relbert-prompt-discrete -m albert-xlarge-v1 --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 3 --export ${PROMPT_DIR}/albert_d923
relbert-prompt-discrete -m albert-xlarge-v1 --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 2 --export ${PROMPT_DIR}/albert_d932
relbert-prompt-discrete -m albert-xlarge-v1 --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 3 --export ${PROMPT_DIR}/albert_d933

relbert-prompt-discrete -m bert-large-cased --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 2 --export ${PROMPT_DIR}/bert_d822
relbert-prompt-discrete -m bert-large-cased --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 2 --export ${PROMPT_DIR}/bert_d832
relbert-prompt-discrete -m bert-large-cased --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 3 --export ${PROMPT_DIR}/bert_d833
relbert-prompt-discrete -m bert-large-cased --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 3 --export ${PROMPT_DIR}/bert_d823
relbert-prompt-discrete -m bert-large-cased --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 2 --export ${PROMPT_DIR}/bert_d922
relbert-prompt-discrete -m bert-large-cased --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 3 --export ${PROMPT_DIR}/bert_d923
relbert-prompt-discrete -m bert-large-cased --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 2 --export ${PROMPT_DIR}/bert_d932
relbert-prompt-discrete -m bert-large-cased --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 3 --export ${PROMPT_DIR}/bert_d933


##############
# continuous #
##############
relbert-prompt-continuous -m roberta-large --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 2 --export ${PROMPT_DIR}/c822
relbert-prompt-continuous -m roberta-large --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 2 --export ${PROMPT_DIR}/c832
relbert-prompt-continuous -m roberta-large --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 3 --export ${PROMPT_DIR}/c833
relbert-prompt-continuous -m roberta-large --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 3 --export ${PROMPT_DIR}/c823
relbert-prompt-continuous -m roberta-large --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 2 --export ${PROMPT_DIR}/c922
relbert-prompt-continuous -m roberta-large --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 3 --export ${PROMPT_DIR}/c923
relbert-prompt-continuous -m roberta-large --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 2 --export ${PROMPT_DIR}/c932
relbert-prompt-continuous -m roberta-large --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 3 --export ${PROMPT_DIR}/c933

relbert-prompt-continuous -m albert-xlarge-v1 --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 2 --export ${PROMPT_DIR}/albert_c822
relbert-prompt-continuous -m albert-xlarge-v1 --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 2 --export ${PROMPT_DIR}/albert_c832
relbert-prompt-continuous -m albert-xlarge-v1 --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 3 --export ${PROMPT_DIR}/albert_c833
relbert-prompt-continuous -m albert-xlarge-v1 --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 3 --export ${PROMPT_DIR}/albert_c823
relbert-prompt-continuous -m albert-xlarge-v1 --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 2 --export ${PROMPT_DIR}/albert_c922
relbert-prompt-continuous -m albert-xlarge-v1 --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 3 --export ${PROMPT_DIR}/albert_c923
relbert-prompt-continuous -m albert-xlarge-v1 --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 2 --export ${PROMPT_DIR}/albert_c932
relbert-prompt-continuous -m albert-xlarge-v1 --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 3 --export ${PROMPT_DIR}/albert_c933

relbert-prompt-continuous -m bert-large-cased --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 2 --export ${PROMPT_DIR}/bert_c822
relbert-prompt-continuous -m bert-large-cased --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 2 --export ${PROMPT_DIR}/bert_c832
relbert-prompt-continuous -m bert-large-cased --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 3 --export ${PROMPT_DIR}/bert_c833
relbert-prompt-continuous -m bert-large-cased --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 3 --export ${PROMPT_DIR}/bert_c823
relbert-prompt-continuous -m bert-large-cased --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 2 --export ${PROMPT_DIR}/bert_c922
relbert-prompt-continuous -m bert-large-cased --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 3 --export ${PROMPT_DIR}/bert_c923
relbert-prompt-continuous -m bert-large-cased --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 2 --export ${PROMPT_DIR}/bert_c932
relbert-prompt-continuous -m bert-large-cased --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 3 --export ${PROMPT_DIR}/bert_c933


