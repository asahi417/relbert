export CKPT_DIR='./relbert_output/ckpt'
export PROMPT_DIR='./relbert_output/prompt_files'
export EVAL_LM='./relbert_output/eval/analogy.csv'

##############
# vanilla LM #
##############
relbert-eval-analogy -m roberta-large -t a --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t b --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t c --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t d --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t e --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t ${PROMPT_DIR}/d832/prompt.json --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t ${PROMPT_DIR}/d922/prompt.json --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t ${PROMPT_DIR}/d833/prompt.json --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t ${PROMPT_DIR}/d823/prompt.json --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t ${PROMPT_DIR}/d932/prompt.json --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t ${PROMPT_DIR}/d933/prompt.json --export-file ${EVAL_LM}

relbert-eval-analogy -m roberta-large -t ${PROMPT_DIR}/d822/prompt.json --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t ${PROMPT_DIR}/d923/prompt.json --export-file ${EVAL_LM}

relbert-eval-analogy -m roberta-large -t ${PROMPT_DIR}/c832/prompt.json --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t ${PROMPT_DIR}/c922/prompt.json --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t ${PROMPT_DIR}/c833/prompt.json --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t ${PROMPT_DIR}/c823/prompt.json --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t ${PROMPT_DIR}/c932/prompt.json --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t ${PROMPT_DIR}/c933/prompt.json --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t ${PROMPT_DIR}/c822/prompt.json --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t ${PROMPT_DIR}/c923/prompt.json --export-file ${EVAL_LM}

###########
# RelBERT #
###########
relbert-eval-analogy -c ${CKPT_DIR}/custom_a --export-file ${EVAL_LM}
relbert-eval-analogy -c ${CKPT_DIR}/custom_b --export-file ${EVAL_LM}
relbert-eval-analogy -c ${CKPT_DIR}/custom_c --export-file ${EVAL_LM}
relbert-eval-analogy -c ${CKPT_DIR}/custom_d --export-file ${EVAL_LM}
relbert-eval-analogy -c ${CKPT_DIR}/custom_e --export-file ${EVAL_LM}
relbert-eval-analogy -c ${CKPT_DIR}/auto_d832 --export-file ${EVAL_LM}
relbert-eval-analogy -c ${CKPT_DIR}/auto_d922 --export-file ${EVAL_LM}
relbert-eval-analogy -c ${CKPT_DIR}/auto_d833 --export-file ${EVAL_LM}
relbert-eval-analogy -c ${CKPT_DIR}/auto_d823 --export-file ${EVAL_LM}
relbert-eval-analogy -c ${CKPT_DIR}/auto_d932 --export-file ${EVAL_LM}
relbert-eval-analogy -c ${CKPT_DIR}/auto_d933 --export-file ${EVAL_LM}

relbert-eval-analogy -c ${CKPT_DIR}/auto_d923 --export-file ${EVAL_LM}
relbert-eval-analogy -c ${CKPT_DIR}/auto_d822 --export-file ${EVAL_LM}

relbert-eval-analogy -c ${CKPT_DIR}/auto_c832 --export-file ${EVAL_LM}
relbert-eval-analogy -c ${CKPT_DIR}/auto_c922 --export-file ${EVAL_LM}
relbert-eval-analogy -c ${CKPT_DIR}/auto_c833 --export-file ${EVAL_LM}
relbert-eval-analogy -c ${CKPT_DIR}/auto_c823 --export-file ${EVAL_LM}
relbert-eval-analogy -c ${CKPT_DIR}/auto_c932 --export-file ${EVAL_LM}
relbert-eval-analogy -c ${CKPT_DIR}/auto_c933 --export-file ${EVAL_LM}

relbert-eval-analogy -c ${CKPT_DIR}/auto_c923 --export-file ${EVAL_LM}
relbert-eval-analogy -c ${CKPT_DIR}/auto_c822 --export-file ${EVAL_LM}


##################
# Classification #
##################
relbert-eval-classification --export-file ./relbert_output/eval/classification.csv