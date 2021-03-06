###################
# Manual template #
###################
relbert-train -m roberta-large -n -p -s -t a --export ./examples/experiments/output/ckpt/roberta_custom_a
relbert-train -m roberta-large -n -p -s -t b --export ./examples/experiments/output/ckpt/roberta_custom_b
relbert-train -m roberta-large -n -p -s -t c --export ./examples/experiments/output/ckpt/roberta_custom_c
relbert-train -m roberta-large -n -p -s -t d --export ./examples/experiments/output/ckpt/roberta_custom_d
relbert-train -m roberta-large -n -p -s -t e --export ./examples/experiments/output/ckpt/roberta_custom_e
relbert-train -m bert-large-cased -n -p -s -t a --export ./examples/experiments/output/ckpt/bert_custom_a
relbert-train -m bert-large-cased -n -p -s -t b --export ./examples/experiments/output/ckpt/bert_custom_b
relbert-train -m bert-large-cased -n -p -s -t c --export ./examples/experiments/output/ckpt/bert_custom_c
relbert-train -m bert-large-cased -n -p -s -t d --export ./examples/experiments/output/ckpt/bert_custom_d
relbert-train -m bert-large-cased -n -p -s -t e --export ./examples/experiments/output/ckpt/bert_custom_e
relbert-train -m albert-xlarge-v1 -n -p -s -t a --export ./examples/experiments/output/ckpt/albert_custom_a -b 32
relbert-train -m albert-xlarge-v1 -n -p -s -t b --export ./examples/experiments/output/ckpt/albert_custom_b -b 32
relbert-train -m albert-xlarge-v1 -n -p -s -t c --export ./examples/experiments/output/ckpt/albert_custom_c -b 32
relbert-train -m albert-xlarge-v1 -n -p -s -t d --export ./examples/experiments/output/ckpt/albert_custom_d -b 32
relbert-train -m albert-xlarge-v1 -n -p -s -t e --export ./examples/experiments/output/ckpt/albert_custom_e -b 32


############
# P-tuning #
############
relbert-train -m roberta-large -n -p -s -t ./examples/experiments/output/prompt_files/roberta_c822/prompt.json --export ./examples/experiments/output/ckpt/roberta_auto_c822
relbert-train -m roberta-large -n -p -s -t ./examples/experiments/output/prompt_files/roberta_c923/prompt.json --export ./examples/experiments/output/ckpt/roberta_auto_c923
relbert-train -m roberta-large -n -p -s -t ./examples/experiments/output/prompt_files/roberta_c833/prompt.json --export ./examples/experiments/output/ckpt/roberta_auto_c833
relbert-train -m roberta-large -n -p -s -t ./examples/experiments/output/prompt_files/roberta_c823/prompt.json --export ./examples/experiments/output/ckpt/roberta_auto_c823
relbert-train -m roberta-large -n -p -s -t ./examples/experiments/output/prompt_files/roberta_c832/prompt.json --export ./examples/experiments/output/ckpt/roberta_auto_c832
relbert-train -m roberta-large -n -p -s -t ./examples/experiments/output/prompt_files/roberta_c922/prompt.json --export ./examples/experiments/output/ckpt/roberta_auto_c922
relbert-train -m roberta-large -n -p -s -t ./examples/experiments/output/prompt_files/roberta_c932/prompt.json --export ./examples/experiments/output/ckpt/roberta_auto_c932
relbert-train -m roberta-large -n -p -s -t ./examples/experiments/output/prompt_files/roberta_c933/prompt.json --export ./examples/experiments/output/ckpt/roberta_auto_c933
relbert-train -m bert-large-cased -n -p -s -t ./examples/experiments/output/prompt_files/bert_c822/prompt.json --export ./examples/experiments/output/ckpt/bert_auto_c822
relbert-train -m bert-large-cased -n -p -s -t ./examples/experiments/output/prompt_files/bert_c923/prompt.json --export ./examples/experiments/output/ckpt/bert_auto_c923
relbert-train -m bert-large-cased -n -p -s -t ./examples/experiments/output/prompt_files/bert_c833/prompt.json --export ./examples/experiments/output/ckpt/bert_auto_c833
relbert-train -m bert-large-cased -n -p -s -t ./examples/experiments/output/prompt_files/bert_c823/prompt.json --export ./examples/experiments/output/ckpt/bert_auto_c823
relbert-train -m bert-large-cased -n -p -s -t ./examples/experiments/output/prompt_files/bert_c832/prompt.json --export ./examples/experiments/output/ckpt/bert_auto_c832
relbert-train -m bert-large-cased -n -p -s -t ./examples/experiments/output/prompt_files/bert_c922/prompt.json --export ./examples/experiments/output/ckpt/bert_auto_c922
relbert-train -m bert-large-cased -n -p -s -t ./examples/experiments/output/prompt_files/bert_c932/prompt.json --export ./examples/experiments/output/ckpt/bert_auto_c932
relbert-train -m bert-large-cased -n -p -s -t ./examples/experiments/output/prompt_files/bert_c933/prompt.json --export ./examples/experiments/output/ckpt/bert_auto_c933
relbert-train -m albert-xlarge-v1 -n -p -s -t ./examples/experiments/output/prompt_files/albert_c822/prompt.json --export ./examples/experiments/output/ckpt/albert_auto_c822 -b 32
relbert-train -m albert-xlarge-v1 -n -p -s -t ./examples/experiments/output/prompt_files/albert_c923/prompt.json --export ./examples/experiments/output/ckpt/albert_auto_c923 -b 32
relbert-train -m albert-xlarge-v1 -n -p -s -t ./examples/experiments/output/prompt_files/albert_c833/prompt.json --export ./examples/experiments/output/ckpt/albert_auto_c833 -b 32
relbert-train -m albert-xlarge-v1 -n -p -s -t ./examples/experiments/output/prompt_files/albert_c823/prompt.json --export ./examples/experiments/output/ckpt/albert_auto_c823 -b 32
relbert-train -m albert-xlarge-v1 -n -p -s -t ./examples/experiments/output/prompt_files/albert_c832/prompt.json --export ./examples/experiments/output/ckpt/albert_auto_c832 -b 32
relbert-train -m albert-xlarge-v1 -n -p -s -t ./examples/experiments/output/prompt_files/albert_c922/prompt.json --export ./examples/experiments/output/ckpt/albert_auto_c922 -b 32
relbert-train -m albert-xlarge-v1 -n -p -s -t ./examples/experiments/output/prompt_files/albert_c932/prompt.json --export ./examples/experiments/output/ckpt/albert_auto_c932 -b 32
relbert-train -m albert-xlarge-v1 -n -p -s -t ./examples/experiments/output/prompt_files/albert_c933/prompt.json --export ./examples/experiments/output/ckpt/albert_auto_c933 -b 32

##############
# AutoPrompt #
##############
relbert-train -m roberta-large -n -p -s -t ./examples/experiments/output/prompt_files/roberta_d822/prompt.json --export ./examples/experiments/output/ckpt/roberta_auto_d822
relbert-train -m roberta-large -n -p -s -t ./examples/experiments/output/prompt_files/roberta_d923/prompt.json --export ./examples/experiments/output/ckpt/roberta_auto_d923
relbert-train -m roberta-large -n -p -s -t ./examples/experiments/output/prompt_files/roberta_d833/prompt.json --export ./examples/experiments/output/ckpt/roberta_auto_d833
relbert-train -m roberta-large -n -p -s -t ./examples/experiments/output/prompt_files/roberta_d823/prompt.json --export ./examples/experiments/output/ckpt/roberta_auto_d823
relbert-train -m roberta-large -n -p -s -t ./examples/experiments/output/prompt_files/roberta_d832/prompt.json --export ./examples/experiments/output/ckpt/roberta_auto_d832
relbert-train -m roberta-large -n -p -s -t ./examples/experiments/output/prompt_files/roberta_d922/prompt.json --export ./examples/experiments/output/ckpt/roberta_auto_d922
relbert-train -m roberta-large -n -p -s -t ./examples/experiments/output/prompt_files/roberta_d932/prompt.json --export ./examples/experiments/output/ckpt/roberta_auto_d932
relbert-train -m roberta-large -n -p -s -t ./examples/experiments/output/prompt_files/roberta_d933/prompt.json --export ./examples/experiments/output/ckpt/roberta_auto_d933
relbert-train -m bert-large-cased -n -p -s -t ./examples/experiments/output/prompt_files/bert_d822/prompt.json --export ./examples/experiments/output/ckpt/bert_auto_d822
relbert-train -m bert-large-cased -n -p -s -t ./examples/experiments/output/prompt_files/bert_d923/prompt.json --export ./examples/experiments/output/ckpt/bert_auto_d923
relbert-train -m bert-large-cased -n -p -s -t ./examples/experiments/output/prompt_files/bert_d833/prompt.json --export ./examples/experiments/output/ckpt/bert_auto_d833
relbert-train -m bert-large-cased -n -p -s -t ./examples/experiments/output/prompt_files/bert_d823/prompt.json --export ./examples/experiments/output/ckpt/bert_auto_d823
relbert-train -m bert-large-cased -n -p -s -t ./examples/experiments/output/prompt_files/bert_d832/prompt.json --export ./examples/experiments/output/ckpt/bert_auto_d832
relbert-train -m bert-large-cased -n -p -s -t ./examples/experiments/output/prompt_files/bert_d922/prompt.json --export ./examples/experiments/output/ckpt/bert_auto_d922
relbert-train -m bert-large-cased -n -p -s -t ./examples/experiments/output/prompt_files/bert_d932/prompt.json --export ./examples/experiments/output/ckpt/bert_auto_d932
relbert-train -m bert-large-cased -n -p -s -t ./examples/experiments/output/prompt_files/bert_d933/prompt.json --export ./examples/experiments/output/ckpt/bert_auto_d933
relbert-train -m albert-xlarge-v1 -n -p -s -t ./examples/experiments/output/prompt_files/albert_d822/prompt.json --export ./examples/experiments/output/ckpt/albert_auto_d822 -b 32
relbert-train -m albert-xlarge-v1 -n -p -s -t ./examples/experiments/output/prompt_files/albert_d923/prompt.json --export ./examples/experiments/output/ckpt/albert_auto_d923 -b 32
relbert-train -m albert-xlarge-v1 -n -p -s -t ./examples/experiments/output/prompt_files/albert_d833/prompt.json --export ./examples/experiments/output/ckpt/albert_auto_d833 -b 32
relbert-train -m albert-xlarge-v1 -n -p -s -t ./examples/experiments/output/prompt_files/albert_d823/prompt.json --export ./examples/experiments/output/ckpt/albert_auto_d823 -b 32
relbert-train -m albert-xlarge-v1 -n -p -s -t ./examples/experiments/output/prompt_files/albert_d832/prompt.json --export ./examples/experiments/output/ckpt/albert_auto_d832 -b 32
relbert-train -m albert-xlarge-v1 -n -p -s -t ./examples/experiments/output/prompt_files/albert_d922/prompt.json --export ./examples/experiments/output/ckpt/albert_auto_d922 -b 32
relbert-train -m albert-xlarge-v1 -n -p -s -t ./examples/experiments/output/prompt_files/albert_d932/prompt.json --export ./examples/experiments/output/ckpt/albert_auto_d932 -b 32
relbert-train -m albert-xlarge-v1 -n -p -s -t ./examples/experiments/output/prompt_files/albert_d933/prompt.json --export ./examples/experiments/output/ckpt/albert_auto_d933 -b 32
