export CKPT_DIR='./relbert_output/ckpt'
export PROMPT_DIR='./relbert_output/prompt_files'

# Train with custom template
#relbert-train -n -p -s -t a --export ${CKPT_DIR}/custom_a
#relbert-train -n -p -s -t b --export ${CKPT_DIR}/custom_b
#relbert-train -n -p -s -t c --export ${CKPT_DIR}/custom_c
#relbert-train -n -p -s -t d --export ${CKPT_DIR}/custom_d
#relbert-train -n -p -s -t e --export ${CKPT_DIR}/custom_e
relbert-train -n -p -s -t a --export ${CKPT_DIR}/custom_a_no_mask --mode average_no_mask
relbert-train -n -p -s -t b --export ${CKPT_DIR}/custom_b_no_mask --mode average_no_mask
relbert-train -n -p -s -t c --export ${CKPT_DIR}/custom_c_no_mask --mode average_no_mask
relbert-train -n -p -s -t d --export ${CKPT_DIR}/custom_d_no_mask --mode average_no_mask
relbert-train -n -p -s -t e --export ${CKPT_DIR}/custom_e_no_mask --mode average_no_mask

# Train with autoprompt
relbert-train -n -p -s -t ${PROMPT_DIR}/822/prompt.json --export ${CKPT_DIR}/autoprompt_822_no_mask --mode average_no_mask
relbert-train -n -p -s -t ${PROMPT_DIR}/833/prompt.json --export ${CKPT_DIR}/autoprompt_833_no_mask --mode average_no_mask
relbert-train -n -p -s -t ${PROMPT_DIR}/823/prompt.json --export ${CKPT_DIR}/autoprompt_823_no_mask --mode average_no_mask
relbert-train -n -p -s -t ${PROMPT_DIR}/832/prompt.json --export ${CKPT_DIR}/autoprompt_832_no_mask --mode average_no_mask
relbert-train -n -p -s -t ${PROMPT_DIR}/922/prompt.json --export ${CKPT_DIR}/autoprompt_922_no_mask --mode average_no_mask
relbert-train -n -p -s -t ${PROMPT_DIR}/923/prompt.json --export ${CKPT_DIR}/autoprompt_923_no_mask --mode average_no_mask

# running
relbert-train -n -p -s -t ${PROMPT_DIR}/932/prompt.json --export ${CKPT_DIR}/autoprompt_932_no_mask --mode average_no_mask
relbert-train -n -p -s -t ${PROMPT_DIR}/933/prompt.json --export ${CKPT_DIR}/autoprompt_933_no_mask --mode average_no_mask
