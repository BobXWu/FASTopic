import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import logging
from typing import List, Callable


def get_top_words(beta, vocab, num_top_words, verbose=False):
    topic_str_list = list()
    for i, topic_dist in enumerate(beta):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(num_top_words + 1):-1]
        topic_str = ' '.join(topic_words)
        topic_str_list.append(topic_str)
        if verbose:
            print('Topic {}: {}'.format(i, topic_str))

    return topic_str_list


class DocEmbedModel:
    def __init__(
        self,
        model: str="all-MiniLM-L6-v2",
        device: str="cpu",
        normalize_embeddings: bool = False,
        verbose: bool = False,
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


class DataIterator:
    def __init__(
        self,
        bow,
        doc_embeddings,
        batch_size,
        device,
        low_memory: bool = False,
        shuffle=True,
    ):
        """
        Args:
            bow: A sparse matrix (scipy.sparse matrix) representing bag-of-words data.
            doc_embeddings: A dense matrix (NumPy array or PyTorch tensor) representing document embeddings.
            batch_size: The size of each batch.
            device: The target device for loading data ('cpu' or 'cuda').
            shuffle: Whether to shuffle the data before batching (default: True).
        """
        self.doc_embeddings = torch.tensor(doc_embeddings, dtype=torch.float32)
        self.low_memory = low_memory
        self.shuffle = shuffle
        self.batch_size = batch_size

        if low_memory:
            self.bow = bow
        else:
            # If memory is enough, convert to tensors.
            self.bow = torch.tensor(bow.toarray(), dtype=torch.float32).to(device)
            self.doc_embeddings = self.doc_embeddings.to(device)

    def __iter__(self):
        """
        Define the iterator logic for batching data.

        Yields:
            A batch of sparse matrix tensors (batch_bow) and dense matrix tensors (batch_doc_embed).
        """
        num_samples = self.bow.shape[0]
        indices = np.arange(num_samples)

        if self.shuffle:
            np.random.shuffle(indices)

        # Generate batches.
        for start_idx in range(0, num_samples, self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]

            if self.low_memory:
                batch_bow = torch.tensor(self.bow[batch_indices].toarray(), dtype=torch.float32)
            else:
                batch_bow = self.bow[batch_indices]

            batch_doc_embed = self.doc_embeddings[batch_indices]
            
            yield batch_bow, batch_doc_embed


class Dataset:
    def __init__(
        self,
        docs: List[str],
        doc_embedder: Callable,
        preprocess: Callable,
        batch_size: int=200,
        device: str="cpu",
        low_memory: bool = False,
        preset_doc_embeddings=None
    ):
        rst = preprocess.preprocess(docs)
        self.train_bow = rst['train_bow']
        self.vocab = rst['vocab']

        self.vocab_size = len(self.vocab)

        if preset_doc_embeddings is None:
            self.doc_embeddings = doc_embedder.encode(docs)
        else:
            self.doc_embeddings = preset_doc_embeddings

        self.doc_embed_size = self.doc_embeddings.shape[1]
        self.dataloader = DataIterator(self.train_bow, self.doc_embeddings, batch_size, device, low_memory)


def check_fitted(model):
    """ Checks if the model was fitted by verifying the presence of self.beta.
    """
    return model.beta is not None


def assert_fitted(model):
    """ assert that the model was fitted.

    Arguments:
        model: FASTopic instance for which the check is performed.

    Returns:
        None

    Raises:
        ValueError: If the beta is not found.
    """
    msg = ("This %(name)s instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

    if not check_fitted(model):
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
