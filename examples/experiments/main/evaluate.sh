##############
# vanilla LM #
##############
relbert-eval --vanilla-lm -c roberta-large -t a --export-file ./relbert_output/eval/analogy.csv
relbert-eval --vanilla-lm -c roberta-large -t b --export-file ./relbert_output/eval/analogy.csv
relbert-eval --vanilla-lm -c roberta-large -t c --export-file ./relbert_output/eval/analogy.csv
relbert-eval --vanilla-lm -c roberta-large -t d --export-file ./relbert_output/eval/analogy.csv
relbert-eval --vanilla-lm -c roberta-large -t e --export-file ./relbert_output/eval/analogy.csv
relbert-eval --vanilla-lm -c roberta-large -t ./relbert_output/prompt_files/roberta_d832/prompt.json --export-file ./relbert_output/eval/analogy.csv
relbert-eval --vanilla-lm -c roberta-large -t ./relbert_output/prompt_files/roberta_d922/prompt.json --export-file ./relbert_output/eval/analogy.csv
relbert-eval --vanilla-lm -c roberta-large -t ./relbert_output/prompt_files/roberta_d833/prompt.json --export-file ./relbert_output/eval/analogy.csv
relbert-eval --vanilla-lm -c roberta-large -t ./relbert_output/prompt_files/roberta_d823/prompt.json --export-file ./relbert_output/eval/analogy.csv
relbert-eval --vanilla-lm -c roberta-large -t ./relbert_output/prompt_files/roberta_d932/prompt.json --export-file ./relbert_output/eval/analogy.csv
relbert-eval --vanilla-lm -c roberta-large -t ./relbert_output/prompt_files/roberta_d933/prompt.json --export-file ./relbert_output/eval/analogy.csv
relbert-eval --vanilla-lm -c roberta-large -t ./relbert_output/prompt_files/roberta_d822/prompt.json --export-file ./relbert_output/eval/analogy.csv
relbert-eval --vanilla-lm -c roberta-large -t ./relbert_output/prompt_files/roberta_d923/prompt.json --export-file ./relbert_output/eval/analogy.csv
relbert-eval --vanilla-lm -c roberta-large -t ./relbert_output/prompt_files/roberta_c832/prompt.json --export-file ./relbert_output/eval/analogy.csv
relbert-eval --vanilla-lm -c roberta-large -t ./relbert_output/prompt_files/roberta_c922/prompt.json --export-file ./relbert_output/eval/analogy.csv
relbert-eval --vanilla-lm -c roberta-large -t ./relbert_output/prompt_files/roberta_c833/prompt.json --export-file ./relbert_output/eval/analogy.csv
relbert-eval --vanilla-lm -c roberta-large -t ./relbert_output/prompt_files/roberta_c823/prompt.json --export-file ./relbert_output/eval/analogy.csv
relbert-eval --vanilla-lm -c roberta-large -t ./relbert_output/prompt_files/roberta_c932/prompt.json --export-file ./relbert_output/eval/analogy.csv
relbert-eval --vanilla-lm -c roberta-large -t ./relbert_output/prompt_files/roberta_c933/prompt.json --export-file ./relbert_output/eval/analogy.csv
relbert-eval --vanilla-lm -c roberta-large -t ./relbert_output/prompt_files/roberta_c822/prompt.json --export-file ./relbert_output/eval/analogy.csv
relbert-eval --vanilla-lm -c roberta-large -t ./relbert_output/prompt_files/roberta_c923/prompt.json --export-file ./relbert_output/eval/analogy.csv

#relbert-eval -c bert-large-cased -t a --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c bert-large-cased -t b --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c bert-large-cased -t c --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c bert-large-cased -t d --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c bert-large-cased -t e --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c bert-cased-large -t ./relbert_output/prompt_files/bert_c922/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c bert-cased-large -t ./relbert_output/prompt_files/bert_c833/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c bert-cased-large -t ./relbert_output/prompt_files/bert_c823/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c bert-cased-large -t ./relbert_output/prompt_files/bert_c932/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c bert-cased-large -t ./relbert_output/prompt_files/bert_c933/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c bert-cased-large -t ./relbert_output/prompt_files/bert_c822/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c bert-cased-large -t ./relbert_output/prompt_files/bert_c923/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c bert-cased-large -t ./relbert_output/prompt_files/bert_c832/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c bert-cased-large -t ./relbert_output/prompt_files/bert_d832/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c bert-cased-large -t ./relbert_output/prompt_files/bert_d922/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c bert-cased-large -t ./relbert_output/prompt_files/bert_d833/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c bert-cased-large -t ./relbert_output/prompt_files/bert_d823/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c bert-cased-large -t ./relbert_output/prompt_files/bert_d932/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c bert-cased-large -t ./relbert_output/prompt_files/bert_d933/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c bert-cased-large -t ./relbert_output/prompt_files/bert_d822/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c bert-cased-large -t ./relbert_output/prompt_files/bert_d923/prompt.json --export-file ./relbert_output/eval/analogy.csv
#
#relbert-eval -c albert-xlarge-v1 -t a --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c albert-xlarge-v1 -t b --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c albert-xlarge-v1 -t c --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c albert-xlarge-v1 -t d --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c albert-xlarge-v1 -t e --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c albert-xlarge-v1 -t ./relbert_output/prompt_files/albert_c922/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c albert-xlarge-v1 -t ./relbert_output/prompt_files/albert_c833/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c albert-xlarge-v1 -t ./relbert_output/prompt_files/albert_c823/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c albert-xlarge-v1 -t ./relbert_output/prompt_files/albert_c932/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c albert-xlarge-v1 -t ./relbert_output/prompt_files/albert_c933/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c albert-xlarge-v1 -t ./relbert_output/prompt_files/albert_c822/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c albert-xlarge-v1 -t ./relbert_output/prompt_files/albert_c923/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c albert-xlarge-v1 -t ./relbert_output/prompt_files/albert_d832/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c albert-xlarge-v1 -t ./relbert_output/prompt_files/albert_d922/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c albert-xlarge-v1 -t ./relbert_output/prompt_files/albert_d833/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c albert-xlarge-v1 -t ./relbert_output/prompt_files/albert_d823/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c albert-xlarge-v1 -t ./relbert_output/prompt_files/albert_d932/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c albert-xlarge-v1 -t ./relbert_output/prompt_files/albert_d933/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c albert-xlarge-v1 -t ./relbert_output/prompt_files/albert_d822/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c albert-xlarge-v1 -t ./relbert_output/prompt_files/albert_d923/prompt.json --export-file ./relbert_output/eval/analogy.csv
#relbert-eval -c albert-xlarge-v1 -t ./relbert_output/prompt_files/albert_c832/prompt.json --export-file ./relbert_output/eval/analogy.csv

###########
# RelBERT #
###########
# analogy
relbert-eval -c 'relbert_output/ckpt/*/*' --export-file ./relbert_output/eval/analogy.csv
# classification
relbert-eval -c 'relbert_output/ckpt/*/*' --type classification --export-file ./relbert_output/eval/relation_classification.csv

