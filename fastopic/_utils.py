import numpy as np
from sentence_transformers import SentenceTransformer

import logging
from typing import List, Tuple, Union, Mapping, Any, Callable, Iterable


class DocEmbedModel:
    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        normalize_embeddings: bool = False,
        device: str = "cpu",
        verbose: bool = False
    ):
        self.verbose = verbose
        self.normalize_embeddings = normalize_embeddings

        if isinstance(model, str):
            self.model = SentenceTransformer(model, device=device)
        else:
            self.model = model

    def encode(self, docs: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            docs,
            show_progress_bar=self.verbose,
            normalize_embeddings=self.normalize_embeddings
        )
        return embeddings


def check_fitted(model):
    """ Checks if the model was fitted by verifying the presence of self.beta

    Arguments:
        model: FASTopic instance for which the check is performed.

    Returns:
        None

    Raises:
        ValueError: If the beta is not found.
    """
    msg = ("This %(name)s instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

    if model.beta is None:
        raise ValueError(msg % {'name': type(model).__name__})


class Logger:
    def __init__(self, level):
        self.logger = logging.getLogger('FASTopic')
        self.set_level(level)
        self._add_handler()
        self.logger.propagate = False

    def info(self, message):
        self.logger.info(f"{message}")

    def warning(self, message):
        self.logger.warning(f"WARNING: {message}")

    def set_level(self, level):
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if level in levels:
            self.logger.setLevel(level)

    def _add_handler(self):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
        self.logger.addHandler(sh)

        # Remove duplicate handlers
        if len(self.logger.handlers) > 1:
            self.logger.handlers = [self.logger.handlers[0]]
