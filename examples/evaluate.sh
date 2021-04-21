export CKPT_DIR='./relbert_output/ckpt'
export PROMPT_DIR='./relbert_output/prompt_files'
export EVAL_LM_CUSTOM='./relbert_output/eval/lm.custom.csv'
export EVAL_LM_AUTO='./relbert_output/eval/lm.autoprompt.csv'
export EVAL_FT_CUSTOM='./relbert_output/eval/finetuned.custom.csv'
export EVAL_FT_AUTO='./relbert_output/eval/finetuned.autoprompt.csv'

# vanilla LM (done)
#relbert-eval -m roberta-large -t a --export-file ${EVAL_LM_CUSTOM}
#relbert-eval -m roberta-large -t b --export-file ${EVAL_LM_CUSTOM}
#relbert-eval -m roberta-large -t c --export-file ${EVAL_LM_CUSTOM}
#relbert-eval -m roberta-large -t d --export-file ${EVAL_LM_CUSTOM}
#relbert-eval -m roberta-large -t e --export-file ${EVAL_LM_CUSTOM}

# vanilla LM (average without mask)
relbert-eval -m roberta-large -t a --export-file ${EVAL_LM_CUSTOM} --mode average_no_mask
relbert-eval -m roberta-large -t b --export-file ${EVAL_LM_CUSTOM} --mode average_no_mask
relbert-eval -m roberta-large -t c --export-file ${EVAL_LM_CUSTOM} --mode average_no_mask
relbert-eval -m roberta-large -t d --export-file ${EVAL_LM_CUSTOM} --mode average_no_mask
relbert-eval -m roberta-large -t e --export-file ${EVAL_LM_CUSTOM} --mode average_no_mask

# finetuned LM
#relbert-eval -c ${CKPT_DIR}/custom_a --export-file ${EVAL_FT_CUSTOM}
#relbert-eval -c ${CKPT_DIR}/custom_b --export-file ${EVAL_FT_CUSTOM}
#relbert-eval -c ${CKPT_DIR}/custom_c --export-file ${EVAL_FT_CUSTOM}
#relbert-eval -c ${CKPT_DIR}/custom_d --export-file ${EVAL_FT_CUSTOM}
#relbert-eval -c ${CKPT_DIR}/custom_e --export-file ${EVAL_FT_CUSTOM}

# finetuned LM (average without mask)
relbert-eval -c ${CKPT_DIR}/custom_a_no_mask --export-file ${EVAL_FT_CUSTOM}
relbert-eval -c ${CKPT_DIR}/custom_b_no_mask --export-file ${EVAL_FT_CUSTOM}
relbert-eval -c ${CKPT_DIR}/custom_c_no_mask --export-file ${EVAL_FT_CUSTOM}
relbert-eval -c ${CKPT_DIR}/custom_d_no_mask --export-file ${EVAL_FT_CUSTOM}
relbert-eval -c ${CKPT_DIR}/custom_e_no_mask --export-file ${EVAL_FT_CUSTOM}

# vanilla LM (autoprompt)
relbert-eval -m roberta-large -t ${PROMPT_DIR}/822/prompt.json --export-file ${EVAL_LM_AUTO} --mode average_no_mask
relbert-eval -m roberta-large -t ${PROMPT_DIR}/833/prompt.json --export-file ${EVAL_LM_AUTO} --mode average_no_mask
relbert-eval -m roberta-large -t ${PROMPT_DIR}/823/prompt.json --export-file ${EVAL_LM_AUTO} --mode average_no_mask
relbert-eval -m roberta-large -t ${PROMPT_DIR}/832/prompt.json --export-file ${EVAL_LM_AUTO} --mode average_no_mask
relbert-eval -m roberta-large -t ${PROMPT_DIR}/922/prompt.json --export-file ${EVAL_LM_AUTO} --mode average_no_mask
relbert-eval -m roberta-large -t ${PROMPT_DIR}/923/prompt.json --export-file ${EVAL_LM_AUTO} --mode average_no_mask

relbert-eval -m roberta-large -t ${PROMPT_DIR}/932/prompt.json --export-file ${EVAL_LM_AUTO} --mode average_no_mask
relbert-eval -m roberta-large -t ${PROMPT_DIR}/933/prompt.json --export-file ${EVAL_LM_AUTO} --mode average_no_mask

# finetuned LM (autoprompt): AutoPrompt shouldn't have mask so use average for all
relbert-eval -c ${CKPT_DIR}/autprompt_822_no_mask --export-file ${EVAL_FT_AUTO}
relbert-eval -c ${CKPT_DIR}/autprompt_833_no_mask --export-file ${EVAL_FT_AUTO}
relbert-eval -c ${CKPT_DIR}/autprompt_823_no_mask --export-file ${EVAL_FT_AUTO}
relbert-eval -c ${CKPT_DIR}/autprompt_832_no_mask --export-file ${EVAL_FT_AUTO}
relbert-eval -c ${CKPT_DIR}/autprompt_922_no_mask --export-file ${EVAL_FT_AUTO}
relbert-eval -c ${CKPT_DIR}/autprompt_923_no_mask --export-file ${EVAL_FT_AUTO}

relbert-eval -c ${CKPT_DIR}/autprompt_932_no_mask --export-file ${EVAL_FT_AUTO}
relbert-eval -c ${CKPT_DIR}/autprompt_933_no_mask --export-file ${EVAL_FT_AUTO}
