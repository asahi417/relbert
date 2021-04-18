export CKPT_DIR='./relbert_output/ckpt'
export PROMPT_DIR='./relbert_output/prompt_files'

# Train with custom template
relbert-train -n -p -s -t a --export ${CKPT_DIR}/custom_a
relbert-train -n -p -s -t b --export ${CKPT_DIR}/custom_b
relbert-train -n -p -s -t c --export ${CKPT_DIR}/custom_c
relbert-train -n -p -s -t d --export ${CKPT_DIR}/custom_d
relbert-train -n -p -s -t e --export ${CKPT_DIR}/custom_e

# Train with autoprompt
relbert-train -n -p -s -t ${PROMPT_DIR}/030/prompt.json --export ${CKPT_DIR}/autoprompt_030
relbert-train -n -p -s -t ${PROMPT_DIR}/040/prompt.json --export ${CKPT_DIR}/autoprompt_040
relbert-train -n -p -s -t ${PROMPT_DIR}/050/prompt.json --export ${CKPT_DIR}/autoprompt_050
relbert-train -n -p -s -t ${PROMPT_DIR}/131/prompt.json --export ${CKPT_DIR}/autoprompt_131
relbert-train -n -p -s -t ${PROMPT_DIR}/141/prompt.json --export ${CKPT_DIR}/autoprompt_141
relbert-train -n -p -s -t ${PROMPT_DIR}/151/prompt.json --export ${CKPT_DIR}/autoprompt_151
relbert-train -n -p -s -t ${PROMPT_DIR}/232/prompt.json --export ${CKPT_DIR}/autoprompt_232
relbert-train -n -p -s -t ${PROMPT_DIR}/242/prompt.json --export ${CKPT_DIR}/autoprompt_242
relbert-train -n -p -s -t ${PROMPT_DIR}/252/prompt.json --export ${CKPT_DIR}/autoprompt_252