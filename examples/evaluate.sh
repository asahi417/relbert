export CKPT_DIR='./relbert_output/ckpt'
export PROMPT_DIR='./relbert_output/prompt_files'
export EVAL_LM_CUSTOM='./relbert_output/eval/lm.custom.csv'
export EVAL_LM_AUTO='./relbert_output/eval/lm.autoprompt.csv'
export EVAL_FT_CUSTOM='./relbert_output/eval/finetuned.custom.csv'
export EVAL_FT_AUTO='./relbert_output/eval/finetuned.autoprompt.csv'

# vanilla LM
relbert-eval -m roberta-large -t a --export-file ${EVAL_LM_CUSTOM}
relbert-eval -m roberta-large -t b --export-file ${EVAL_LM_CUSTOM}
relbert-eval -m roberta-large -t c --export-file ${EVAL_LM_CUSTOM}
relbert-eval -m roberta-large -t d --export-file ${EVAL_LM_CUSTOM}
relbert-eval -m roberta-large -t e --export-file ${EVAL_LM_CUSTOM}

# vanilla LM (average without mask)
relbert-eval -m roberta-large -t a --export-file ${EVAL_LM_CUSTOM} --mode average_no_mask
relbert-eval -m roberta-large -t b --export-file ${EVAL_LM_CUSTOM} --mode average_no_mask
relbert-eval -m roberta-large -t c --export-file ${EVAL_LM_CUSTOM} --mode average_no_mask
relbert-eval -m roberta-large -t d --export-file ${EVAL_LM_CUSTOM} --mode average_no_mask
relbert-eval -m roberta-large -t e --export-file ${EVAL_LM_CUSTOM} --mode average_no_mask

# finetuned LM
relbert-eval -c ${CKPT_DIR}/custom_a --export-file ${EVAL_FT_CUSTOM}
relbert-eval -c ${CKPT_DIR}/custom_b --export-file ${EVAL_FT_CUSTOM}
relbert-eval -c ${CKPT_DIR}/custom_c --export-file ${EVAL_FT_CUSTOM}
relbert-eval -c ${CKPT_DIR}/custom_d --export-file ${EVAL_FT_CUSTOM}
relbert-eval -c ${CKPT_DIR}/custom_e --export-file ${EVAL_FT_CUSTOM}

relbert-eval -c ${CKPT_DIR}/custom_d_10 --export-file ${EVAL_FT_CUSTOM}

# finetuned LM (average without mask)
relbert-eval -c ${CKPT_DIR}/custom_a_no_mask --export-file ${EVAL_FT_CUSTOM}
relbert-eval -c ${CKPT_DIR}/custom_b_no_mask --export-file ${EVAL_FT_CUSTOM}
relbert-eval -c ${CKPT_DIR}/custom_c_no_mask --export-file ${EVAL_FT_CUSTOM}
relbert-eval -c ${CKPT_DIR}/custom_d_no_mask --export-file ${EVAL_FT_CUSTOM}
relbert-eval -c ${CKPT_DIR}/custom_e_no_mask --export-file ${EVAL_FT_CUSTOM}

relbert-eval -c ${CKPT_DIR}/custom_d_no_mask_10 --export-file ${EVAL_FT_CUSTOM}

# vanilla LM (autoprompt): AutoPrompt shouldn't have mask so use average for all
relbert-eval -m roberta-large -t ${PROMPT_DIR}/030/prompt.json --export-file ${EVAL_LM_AUTO}
relbert-eval -m roberta-large -t ${PROMPT_DIR}/040/prompt.json --export-file ${EVAL_LM_AUTO}
relbert-eval -m roberta-large -t ${PROMPT_DIR}/050/prompt.json --export-file ${EVAL_LM_AUTO}
relbert-eval -m roberta-large -t ${PROMPT_DIR}/131/prompt.json --export-file ${EVAL_LM_AUTO}
relbert-eval -m roberta-large -t ${PROMPT_DIR}/141/prompt.json --export-file ${EVAL_LM_AUTO}
relbert-eval -m roberta-large -t ${PROMPT_DIR}/151/prompt.json --export-file ${EVAL_LM_AUTO}
relbert-eval -m roberta-large -t ${PROMPT_DIR}/232/prompt.json --export-file ${EVAL_LM_AUTO}
relbert-eval -m roberta-large -t ${PROMPT_DIR}/242/prompt.json --export-file ${EVAL_LM_AUTO}
relbert-eval -m roberta-large -t ${PROMPT_DIR}/252/prompt.json --export-file ${EVAL_LM_AUTO}

# finetuned LM (autoprompt): AutoPrompt shouldn't have mask so use average for all
relbert-eval -c ${CKPT_DIR}/autprompt_030 --export-file ${EVAL_FT_AUTO}
relbert-eval -c ${CKPT_DIR}/autprompt_040 --export-file ${EVAL_FT_AUTO}
relbert-eval -c ${CKPT_DIR}/autprompt_050 --export-file ${EVAL_FT_AUTO}
relbert-eval -c ${CKPT_DIR}/autprompt_131 --export-file ${EVAL_FT_AUTO}
relbert-eval -c ${CKPT_DIR}/autprompt_141 --export-file ${EVAL_FT_AUTO}
relbert-eval -c ${CKPT_DIR}/autprompt_151 --export-file ${EVAL_FT_AUTO}
relbert-eval -c ${CKPT_DIR}/autprompt_232 --export-file ${EVAL_FT_AUTO}
relbert-eval -c ${CKPT_DIR}/autprompt_242 --export-file ${EVAL_FT_AUTO}
relbert-eval -c ${CKPT_DIR}/autprompt_252 --export-file ${EVAL_FT_AUTO}

