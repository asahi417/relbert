export PROMPT_DIR='./relbert_output/prompt_files'

############
# discrete #
############
relbert-prompt-discrete --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 2 --export ${PROMPT_DIR}/d822
relbert-prompt-discrete --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 2 --export ${PROMPT_DIR}/d832
relbert-prompt-discrete --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 3 --export ${PROMPT_DIR}/d833
relbert-prompt-discrete --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 3 --export ${PROMPT_DIR}/d823
relbert-prompt-discrete --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 2 --export ${PROMPT_DIR}/d922
relbert-prompt-discrete --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 3 --export ${PROMPT_DIR}/d923
relbert-prompt-discrete --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 2 --export ${PROMPT_DIR}/d932
relbert-prompt-discrete --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 3 --export ${PROMPT_DIR}/d933

##############
# continuous #
##############
relbert-prompt-continuous --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 2 --export ${PROMPT_DIR}/c822
relbert-prompt-continuous --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 2 --export ${PROMPT_DIR}/c832
relbert-prompt-continuous --n-trigger-b 8 --n-trigger-i 3 --n-trigger-e 3 --export ${PROMPT_DIR}/c833
relbert-prompt-continuous --n-trigger-b 8 --n-trigger-i 2 --n-trigger-e 3 --export ${PROMPT_DIR}/c823
relbert-prompt-continuous --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 2 --export ${PROMPT_DIR}/c922
relbert-prompt-continuous --n-trigger-b 9 --n-trigger-i 2 --n-trigger-e 3 --export ${PROMPT_DIR}/c923
relbert-prompt-continuous --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 2 --export ${PROMPT_DIR}/c932
relbert-prompt-continuous --n-trigger-b 9 --n-trigger-i 3 --n-trigger-e 3 --export ${PROMPT_DIR}/c933

