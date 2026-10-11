"""Text cleaning shared by training and serving."""

from functools import lru_cache

from src.config import BASE_MODEL


@lru_cache(maxsize=1)
def _get_preprocessor(model_name: str):
    # Imported here so that `import src.text` stays instant.
    from arabert.preprocess import ArabertPreprocessor

    return ArabertPreprocessor(model_name=model_name)


def clean_text(text: str, model_name: str = BASE_MODEL) -> str:
    """Clean one Arabic review the way AraBERT's authors cleaned its training data."""
    return _get_preprocessor(model_name).preprocess(text)
