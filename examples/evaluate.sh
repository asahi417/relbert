export CKPT_DIR='./relbert_output/ckpt'
export PROMPT_DIR='./relbert_output/prompt_files'
export EVAL_LM_CUSTOM='./relbert_output/eval/lm.custom.csv'
export EVAL_LM_AUTO='./relbert_output/eval/lm.autoprompt.csv'
export EVAL_FT_CUSTOM='./relbert_output/eval/finetuned.custom.csv'
export EVAL_FT_AUTO='./relbert_output/eval/finetuned.autoprompt.csv'

relbert-eval --debug -m roberta-large -t a --export-file ${EVAL_LM_CUSTOM}
relbert-eval --debug -m roberta-large -t b --export-file ${EVAL_LM_CUSTOM}
relbert-eval --debug -m roberta-large -t c --export-file ${EVAL_LM_CUSTOM}
relbert-eval --debug -m roberta-large -t d --export-file ${EVAL_LM_CUSTOM}
relbert-eval --debug -m roberta-large -t e --export-file ${EVAL_LM_CUSTOM}
#relbert-eval --debug -m roberta-large -t f --export-file ${EVAL_LM_CUSTOM}
#relbert-eval --debug -m roberta-large -t g --export-file ${EVAL_LM_CUSTOM}
#relbert-eval --debug -m roberta-large -t h --export-file ${EVAL_LM_CUSTOM}
#relbert-eval --debug -m roberta-large -t i --export-file ${EVAL_LM_CUSTOM}
relbert-eval --debug -m roberta-large -t ${PROMPT_DIR}/030 --export-file ${EVAL_LM_AUTO}
relbert-eval --debug -m roberta-large -t ${PROMPT_DIR}/040 --export-file ${EVAL_LM_AUTO}
relbert-eval --debug -m roberta-large -t ${PROMPT_DIR}/050 --export-file ${EVAL_LM_AUTO}
relbert-eval --debug -m roberta-large -t ${PROMPT_DIR}/131 --export-file ${EVAL_LM_AUTO}
relbert-eval --debug -m roberta-large -t ${PROMPT_DIR}/141 --export-file ${EVAL_LM_AUTO}
relbert-eval --debug -m roberta-large -t ${PROMPT_DIR}/151 --export-file ${EVAL_LM_AUTO}
relbert-eval --debug -m roberta-large -t ${PROMPT_DIR}/232 --export-file ${EVAL_LM_AUTO}
relbert-eval --debug -m roberta-large -t ${PROMPT_DIR}/242 --export-file ${EVAL_LM_AUTO}
relbert-eval --debug -m roberta-large -t ${PROMPT_DIR}/252 --export-file ${EVAL_LM_AUTO}

relbert-eval -c ${CKPT_DIR}/custom_a --export-file ${EVAL_FT_CUSTOM}
relbert-eval -c ${CKPT_DIR}/custom_b --export-file ${EVAL_FT_CUSTOM}
relbert-eval -c ${CKPT_DIR}/custom_c --export-file ${EVAL_FT_CUSTOM}
relbert-eval -c ${CKPT_DIR}/custom_d --export-file ${EVAL_FT_CUSTOM}
relbert-eval -c ${CKPT_DIR}/custom_e --export-file ${EVAL_FT_CUSTOM}
#relbert-eval -c ${CKPT_DIR}/custom_f --export-file ${EVAL_FT_CUSTOM}
#relbert-eval -c ${CKPT_DIR}/custom_g --export-file ${EVAL_FT_CUSTOM}
#relbert-eval -c ${CKPT_DIR}/custom_h --export-file ${EVAL_FT_CUSTOM}
#relbert-eval -c ${CKPT_DIR}/custom_i --export-file ${EVAL_FT_CUSTOM}
relbert-eval -c ${CKPT_DIR}/autprompt_030 --export-file ${EVAL_FT_AUTO}
relbert-eval -c ${CKPT_DIR}/autprompt_040 --export-file ${EVAL_FT_AUTO}
relbert-eval -c ${CKPT_DIR}/autprompt_050 --export-file ${EVAL_FT_AUTO}
relbert-eval -c ${CKPT_DIR}/autprompt_131 --export-file ${EVAL_FT_AUTO}
relbert-eval -c ${CKPT_DIR}/autprompt_141 --export-file ${EVAL_FT_AUTO}
relbert-eval -c ${CKPT_DIR}/autprompt_151 --export-file ${EVAL_FT_AUTO}
relbert-eval -c ${CKPT_DIR}/autprompt_232 --export-file ${EVAL_FT_AUTO}
relbert-eval -c ${CKPT_DIR}/autprompt_242 --export-file ${EVAL_FT_AUTO}
relbert-eval -c ${CKPT_DIR}/autprompt_252 --export-file ${EVAL_FT_AUTO}

