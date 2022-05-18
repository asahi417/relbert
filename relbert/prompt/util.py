import transformers


def load_language_model(model_name, cache_dir: str = None):
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    except ValueError:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
    try:
        config = transformers.AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    except ValueError:
        config = transformers.AutoConfig.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
    try:
        model = transformers.AutoModel.from_pretrained(model_name, config=config, cache_dir=cache_dir)
    except ValueError:
        model = transformers.AutoModel.from_pretrained(model_name, config=config, cache_dir=cache_dir, local_files_only=True)
    return tokenizer, model, config
