export PROMPT_DIR='./relbert_output/prompt_files'

relbert-prompt --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 2 --export ${PROMPT_DIR}/822 --n-iteration 50
relbert-prompt --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 2 --export ${PROMPT_DIR}/832 --n-iteration 50
relbert-prompt --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 3 --export ${PROMPT_DIR}/833 --n-iteration 50
relbert-prompt --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 2 --export ${PROMPT_DIR}/922 --n-iteration 50
relbert-prompt --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 3 --export ${PROMPT_DIR}/923 --n-iteration 50
relbert-prompt --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 3 --export ${PROMPT_DIR}/823 --n-iteration 50
relbert-prompt --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 2 --export ${PROMPT_DIR}/932 --n-iteration 50
relbert-prompt --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 3 --export ${PROMPT_DIR}/933 --n-iteration 50


