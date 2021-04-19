export PROMPT_DIR='./relbert_output/prompt_files'

relbert-prompt --n-trigger-b 0 --n-trigger-i 3 --n-trigger-e 0 --export ${PROMPT_DIR}/030
relbert-prompt --n-trigger-b 0 --n-trigger-i 4 --n-trigger-e 0 --export ${PROMPT_DIR}/040
relbert-prompt --n-trigger-b 0 --n-trigger-i 5 --n-trigger-e 0 --export ${PROMPT_DIR}/050
relbert-prompt --n-trigger-b 1 --n-trigger-i 3 --n-trigger-e 1 --export ${PROMPT_DIR}/131
relbert-prompt --n-trigger-b 1 --n-trigger-i 4 --n-trigger-e 1 --export ${PROMPT_DIR}/141
relbert-prompt --n-trigger-b 1 --n-trigger-i 5 --n-trigger-e 1 --export ${PROMPT_DIR}/151
relbert-prompt --n-trigger-b 2 --n-trigger-i 3 --n-trigger-e 2 --export ${PROMPT_DIR}/232
relbert-prompt --n-trigger-b 2 --n-trigger-i 4 --n-trigger-e 2 --export ${PROMPT_DIR}/242
# on stone server
relbert-prompt --n-trigger-b 2 --n-trigger-i 5 --n-trigger-e 2 --export ${PROMPT_DIR}/252

# not yet
relbert-prompt --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 2 --export ${PROMPT_DIR}/822 --n-iteration 50
relbert-prompt --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 2 --export ${PROMPT_DIR}/832 --n-iteration 50
relbert-prompt --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 3 --export ${PROMPT_DIR}/833 --n-iteration 50
relbert-prompt --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 2 --export ${PROMPT_DIR}/922 --n-iteration 50
relbert-prompt --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 3 --export ${PROMPT_DIR}/923 --n-iteration 50

# running
relbert-prompt --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 3 --export ${PROMPT_DIR}/823 --n-iteration 50
relbert-prompt --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 2 --export ${PROMPT_DIR}/932 --n-iteration 50
relbert-prompt --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 3 --export ${PROMPT_DIR}/933 --n-iteration 50


