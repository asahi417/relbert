export CKPT_DIR='./relbert_output/ckpt'
export PROMPT_DIR='./relbert_output/prompt_files'

# Train with custom template
relbert-train -n -p -s -t a --export ${CKPT_DIR}/custom_a -b 16 -e 5
relbert-train -n -p -s -t b --export ${CKPT_DIR}/custom_b -b 16 -e 5
relbert-train -n -p -s -t c --export ${CKPT_DIR}/custom_c -b 16 -e 5
relbert-train -n -p -s -t d --export ${CKPT_DIR}/custom_d -b 16 -e 5
relbert-train -n -p -s -t e --export ${CKPT_DIR}/custom_e -b 16 -e 5

relbert-train -n -p -s -t a --export ${CKPT_DIR}/custom_a_no_mask --mode average_no_mask -b 16 -e 5
relbert-train -n -p -s -t b --export ${CKPT_DIR}/custom_b_no_mask --mode average_no_mask -b 16 -e 5
relbert-train -n -p -s -t c --export ${CKPT_DIR}/custom_c_no_mask --mode average_no_mask -b 16 -e 5
relbert-train -n -p -s -t d --export ${CKPT_DIR}/custom_d_no_mask --mode average_no_mask -b 16 -e 5
relbert-train -n -p -s -t e --export ${CKPT_DIR}/custom_e_no_mask --mode average_no_mask -b 16 -e 5

# Train with custom template (more data)
relbert-train -n -p -s -t a --export ${CKPT_DIR}/custom_a_10 --n-sample 10 -b 8 -e 5
relbert-train -n -p -s -t b --export ${CKPT_DIR}/custom_b_10 --n-sample 10 -b 8 -e 5
relbert-train -n -p -s -t c --export ${CKPT_DIR}/custom_c_10 --n-sample 10 -b 8 -e 5
relbert-train -n -p -s -t d --export ${CKPT_DIR}/custom_d_10 --n-sample 10 -b 8 -e 5
relbert-train -n -p -s -t e --export ${CKPT_DIR}/custom_e_10 --n-sample 10 -b 8 -e 5

relbert-train -n -p -s -t a --export ${CKPT_DIR}/custom_a_no_mask_10 --mode average_no_mask --n-sample 10 -b 8 -e 5
relbert-train -n -p -s -t b --export ${CKPT_DIR}/custom_b_no_mask_10 --mode average_no_mask --n-sample 10 -b 8 -e 5
relbert-train -n -p -s -t c --export ${CKPT_DIR}/custom_c_no_mask_10 --mode average_no_mask --n-sample 10 -b 8 -e 5
relbert-train -n -p -s -t d --export ${CKPT_DIR}/custom_d_no_mask_10 --mode average_no_mask --n-sample 10 -b 8 -e 5
relbert-train -n -p -s -t e --export ${CKPT_DIR}/custom_e_no_mask_10 --mode average_no_mask --n-sample 10 -b 8 -e 5

# Train with autoprompt
relbert-train -n -p -s -t ${PROMPT_DIR}/030/prompt.json --export ${CKPT_DIR}/autoprompt_030
relbert-train -n -p -s -t ${PROMPT_DIR}/141/prompt.json --export ${CKPT_DIR}/autoprompt_141
relbert-train -n -p -s -t ${PROMPT_DIR}/040/prompt.json --export ${CKPT_DIR}/autoprompt_040
relbert-train -n -p -s -t ${PROMPT_DIR}/050/prompt.json --export ${CKPT_DIR}/autoprompt_050
relbert-train -n -p -s -t ${PROMPT_DIR}/131/prompt.json --export ${CKPT_DIR}/autoprompt_131
relbert-train -n -p -s -t ${PROMPT_DIR}/151/prompt.json --export ${CKPT_DIR}/autoprompt_151
relbert-train -n -p -s -t ${PROMPT_DIR}/232/prompt.json --export ${CKPT_DIR}/autoprompt_232
relbert-train -n -p -s -t ${PROMPT_DIR}/242/prompt.json --export ${CKPT_DIR}/autoprompt_242
relbert-train -n -p -s -t ${PROMPT_DIR}/252/prompt.json --export ${CKPT_DIR}/autoprompt_252