import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM


def import_tokenizer(conf):
    model_id = conf["model"]
    # Persist tokenizer under a single project-level "model" folder
    local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))

    os.makedirs(local_dir, exist_ok=True)

    # If tokenizer already saved locally, load from there
    if os.path.isdir(local_dir) and os.path.isfile(os.path.join(local_dir, "tokenizer_config.json")):
        return AutoTokenizer.from_pretrained(local_dir, local_files_only=True)

    # Else download then save for future runs
    tokenizer = AutoTokenizer.from_pretrained(model_id, resume_download=True)
    tokenizer.save_pretrained(local_dir)
    return tokenizer


def import_model(conf):
    model_id = conf["model"]
    # Persist model weights under a single project-level "model" folder
    local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))

    os.makedirs(local_dir, exist_ok=True)

    # If model already saved locally, load from there
    if os.path.isdir(local_dir) and (
        os.path.isfile(os.path.join(local_dir, "pytorch_model.bin"))
        or os.path.isfile(os.path.join(local_dir, "model.safetensors"))
    ):
        if conf["architecture"] == "encoder_decoder":
            return AutoModelForSeq2SeqLM.from_pretrained(local_dir, local_files_only=True)
        else:
            return AutoModelForCausalLM.from_pretrained(local_dir, local_files_only=True)

    # Else download from hub then save for future runs
    if conf["architecture"] == "encoder_decoder":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, resume_download=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, resume_download=True)

    model.save_pretrained(local_dir)
    return model



