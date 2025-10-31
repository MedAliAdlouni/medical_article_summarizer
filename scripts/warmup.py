"""
Warm up the container by downloading and saving the tokenizer/model
into the project-level model directory used by src.core.*
"""

from src.core.config import conf
from src.core.models import import_model, import_tokenizer


def main() -> None:
    tok = import_tokenizer(conf)
    mdl = import_model(conf)
    _ = (tok is not None) and (mdl is not None)


if __name__ == "__main__":
    main()


