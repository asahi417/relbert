export CKPT_DIR='./relbert_output/ckpt'
export PROMPT_DIR='./relbert_output/prompt_files'
export EVAL_LM='./relbert_output/eval/analogy.lm.csv'
export EVAL_FT_CUSTOM='./relbert_output/eval/finetuned.custom.csv'
export EVAL_FT_AUTO='./relbert_output/eval/finetuned.autoprompt.csv'

##############
# vanilla LM #
##############
relbert-eval-analogy -m roberta-large -t a --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t b --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t c --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t d --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t e --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t ${PROMPT_DIR}/d822/prompt.json --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t ${PROMPT_DIR}/d923/prompt.json --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t ${PROMPT_DIR}/d832/prompt.json --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t ${PROMPT_DIR}/d922/prompt.json --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t ${PROMPT_DIR}/d833/prompt.json --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t ${PROMPT_DIR}/d823/prompt.json --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t ${PROMPT_DIR}/d932/prompt.json --export-file ${EVAL_LM}
relbert-eval-analogy -m roberta-large -t ${PROMPT_DIR}/d933/prompt.json --export-file ${EVAL_LM}


# finetuned LM
relbert-eval-analogy -c ${CKPT_DIR}/custom_a_no_mask --export-file ${EVAL_FT_CUSTOM}
relbert-eval-analogy -c ${CKPT_DIR}/custom_b_no_mask --export-file ${EVAL_FT_CUSTOM}
relbert-eval-analogy -c ${CKPT_DIR}/custom_c_no_mask --export-file ${EVAL_FT_CUSTOM}
relbert-eval-analogy -c ${CKPT_DIR}/custom_d_no_mask --export-file ${EVAL_FT_CUSTOM}
relbert-eval-analogy -c ${CKPT_DIR}/custom_e_no_mask --export-file ${EVAL_FT_CUSTOM}


# finetuned LM (autoprompt): AutoPrompt shouldn't have mask so use average for all
relbert-eval-analogy -c ${CKPT_DIR}/autoprompt_822_no_mask --export-file ${EVAL_FT_AUTO}
relbert-eval-analogy -c ${CKPT_DIR}/autoprompt_923_no_mask --export-file ${EVAL_FT_AUTO}

relbert-eval-analogy -c ${CKPT_DIR}/autoprompt_832_no_mask --export-file ${EVAL_FT_AUTO}
relbert-eval-analogy -c ${CKPT_DIR}/autoprompt_922_no_mask --export-file ${EVAL_FT_AUTO}
relbert-eval-analogy -c ${CKPT_DIR}/autoprompt_833_no_mask --export-file ${EVAL_FT_AUTO}
relbert-eval-analogy -c ${CKPT_DIR}/autoprompt_823_no_mask --export-file ${EVAL_FT_AUTO}
relbert-eval-analogy -c ${CKPT_DIR}/autoprompt_932_no_mask --export-file ${EVAL_FT_AUTO}
relbert-eval-analogy -c ${CKPT_DIR}/autoprompt_933_no_mask --export-file ${EVAL_FT_AUTO}
