##############
# continuous #
##############
relbert-prompt-continuous -n -m roberta-large --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 2 --export ./relbert_output/prompt_files/roberta_c822
relbert-prompt-continuous -n -m roberta-large --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 2 --export ./relbert_output/prompt_files/roberta_c832
relbert-prompt-continuous -n -m roberta-large --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 3 --export ./relbert_output/prompt_files/roberta_c833
relbert-prompt-continuous -n -m roberta-large --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 3 --export ./relbert_output/prompt_files/roberta_c823
relbert-prompt-continuous -n -m roberta-large --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 2 --export ./relbert_output/prompt_files/roberta_c922
relbert-prompt-continuous -n -m roberta-large --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 3 --export ./relbert_output/prompt_files/roberta_c923
relbert-prompt-continuous -n -m roberta-large --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 2 --export ./relbert_output/prompt_files/roberta_c932
relbert-prompt-continuous -n -m roberta-large --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 3 --export ./relbert_output/prompt_files/roberta_c933
relbert-prompt-continuous -n -m bert-large-cased --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 2 --export ./relbert_output/prompt_files/bert_c822
relbert-prompt-continuous -n -m bert-large-cased --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 2 --export ./relbert_output/prompt_files/bert_c832
relbert-prompt-continuous -n -m bert-large-cased --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 3 --export ./relbert_output/prompt_files/bert_c833
relbert-prompt-continuous -n -m bert-large-cased --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 3 --export ./relbert_output/prompt_files/bert_c823
relbert-prompt-continuous -n -m bert-large-cased --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 2 --export ./relbert_output/prompt_files/bert_c922
relbert-prompt-continuous -n -m bert-large-cased --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 3 --export ./relbert_output/prompt_files/bert_c923
relbert-prompt-continuous -n -m bert-large-cased --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 2 --export ./relbert_output/prompt_files/bert_c932
relbert-prompt-continuous -n -m bert-large-cased --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 3 --export ./relbert_output/prompt_files/bert_c933
relbert-prompt-continuous -n -m albert-xlarge-v1 --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 2 --export ./relbert_output/prompt_files/albert_c822 -b 32
relbert-prompt-continuous -n -m albert-xlarge-v1 --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 2 --export ./relbert_output/prompt_files/albert_c832 -b 32
relbert-prompt-continuous -n -m albert-xlarge-v1 --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 3 --export ./relbert_output/prompt_files/albert_c833 -b 32
relbert-prompt-continuous -n -m albert-xlarge-v1 --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 3 --export ./relbert_output/prompt_files/albert_c823 -b 32
relbert-prompt-continuous -n -m albert-xlarge-v1 --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 2 --export ./relbert_output/prompt_files/albert_c922 -b 32
relbert-prompt-continuous -n -m albert-xlarge-v1 --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 3 --export ./relbert_output/prompt_files/albert_c923 -b 32
relbert-prompt-continuous -n -m albert-xlarge-v1 --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 2 --export ./relbert_output/prompt_files/albert_c932 -b 32
relbert-prompt-continuous -n -m albert-xlarge-v1 --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 3 --export ./relbert_output/prompt_files/albert_c933 -b 32



############
# discrete #
############
relbert-prompt-discrete -m roberta-large --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 2 --export ./relbert_output/prompt_files/roberta_d822
relbert-prompt-discrete -m roberta-large --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 2 --export ./relbert_output/prompt_files/roberta_d832
relbert-prompt-discrete -m roberta-large --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 3 --export ./relbert_output/prompt_files/roberta_d833
relbert-prompt-discrete -m roberta-large --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 3 --export ./relbert_output/prompt_files/roberta_d823
relbert-prompt-discrete -m roberta-large --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 2 --export ./relbert_output/prompt_files/roberta_d922
relbert-prompt-discrete -m roberta-large --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 3 --export ./relbert_output/prompt_files/roberta_d923
relbert-prompt-discrete -m roberta-large --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 2 --export ./relbert_output/prompt_files/roberta_d932
relbert-prompt-discrete -m roberta-large --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 3 --export ./relbert_output/prompt_files/roberta_d933

relbert-prompt-discrete -m albert-xlarge-v1 --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 2 --export ./relbert_output/prompt_files/albert_d822
relbert-prompt-discrete -m albert-xlarge-v1 --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 2 --export ./relbert_output/prompt_files/albert_d832
relbert-prompt-discrete -m albert-xlarge-v1 --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 3 --export ./relbert_output/prompt_files/albert_d833
relbert-prompt-discrete -m albert-xlarge-v1 --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 3 --export ./relbert_output/prompt_files/albert_d823
relbert-prompt-discrete -m albert-xlarge-v1 --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 2 --export ./relbert_output/prompt_files/albert_d922
relbert-prompt-discrete -m albert-xlarge-v1 --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 3 --export ./relbert_output/prompt_files/albert_d923
relbert-prompt-discrete -m albert-xlarge-v1 --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 2 --export ./relbert_output/prompt_files/albert_d932
relbert-prompt-discrete -m albert-xlarge-v1 --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 3 --export ./relbert_output/prompt_files/albert_d933

relbert-prompt-discrete -m bert-large-cased --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 2 --export ./relbert_output/prompt_files/bert_d822
relbert-prompt-discrete -m bert-large-cased --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 2 --export ./relbert_output/prompt_files/bert_d832
relbert-prompt-discrete -m bert-large-cased --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 3 --export ./relbert_output/prompt_files/bert_d833

relbert-prompt-discrete -m bert-large-cased --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 3 --export ./relbert_output/prompt_files/bert_d823
relbert-prompt-discrete -m bert-large-cased --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 2 --export ./relbert_output/prompt_files/bert_d922
relbert-prompt-discrete -m bert-large-cased --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 3 --export ./relbert_output/prompt_files/bert_d923
relbert-prompt-discrete -m bert-large-cased --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 2 --export ./relbert_output/prompt_files/bert_d932
relbert-prompt-discrete -m bert-large-cased --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 3 --export ./relbert_output/prompt_files/bert_d933
