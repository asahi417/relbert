export CKPT_DIR='./relbert_output/ckpt'
export PROMPT_DIR='./relbert_output/prompt_files'

# Train with custom template
relbert-train -n -p -s -t a --export ${CKPT_DIR}/custom_a
relbert-train -n -p -s -t b --export ${CKPT_DIR}/custom_b
relbert-train -n -p -s -t c --export ${CKPT_DIR}/custom_c
relbert-train -n -p -s -t d --export ${CKPT_DIR}/custom_d
relbert-train -n -p -s -t e --export ${CKPT_DIR}/custom_e

# Train with discrete tuned prompt
relbert-train -n -p -s -t ${PROMPT_DIR}/d822/prompt.json --export ${CKPT_DIR}/auto_d822
relbert-train -n -p -s -t ${PROMPT_DIR}/d923/prompt.json --export ${CKPT_DIR}/auto_d923

relbert-train -n -p -s -t ${PROMPT_DIR}/d833/prompt.json --export ${CKPT_DIR}/auto_d833
relbert-train -n -p -s -t ${PROMPT_DIR}/d823/prompt.json --export ${CKPT_DIR}/auto_d823
relbert-train -n -p -s -t ${PROMPT_DIR}/d832/prompt.json --export ${CKPT_DIR}/auto_d832
relbert-train -n -p -s -t ${PROMPT_DIR}/d922/prompt.json --export ${CKPT_DIR}/auto_d922
relbert-train -n -p -s -t ${PROMPT_DIR}/d932/prompt.json --export ${CKPT_DIR}/auto_d932
relbert-train -n -p -s -t ${PROMPT_DIR}/d933/prompt.json --export ${CKPT_DIR}/auto_d933

# Train with continuous tuned prompt
relbert-train -n -p -s -t ${PROMPT_DIR}/c822/prompt.json --export ${CKPT_DIR}/auto_c822
relbert-train -n -p -s -t ${PROMPT_DIR}/c923/prompt.json --export ${CKPT_DIR}/auto_c923
relbert-train -n -p -s -t ${PROMPT_DIR}/c833/prompt.json --export ${CKPT_DIR}/auto_c833
relbert-train -n -p -s -t ${PROMPT_DIR}/c823/prompt.json --export ${CKPT_DIR}/auto_c823
relbert-train -n -p -s -t ${PROMPT_DIR}/c832/prompt.json --export ${CKPT_DIR}/auto_c832
relbert-train -n -p -s -t ${PROMPT_DIR}/c922/prompt.json --export ${CKPT_DIR}/auto_c922
relbert-train -n -p -s -t ${PROMPT_DIR}/c932/prompt.json --export ${CKPT_DIR}/auto_c932
relbert-train -n -p -s -t ${PROMPT_DIR}/c933/prompt.json --export ${CKPT_DIR}/auto_c933
