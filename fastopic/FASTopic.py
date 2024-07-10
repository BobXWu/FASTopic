import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

import torch
from topmost.utils._utils import get_top_words
from topmost.data import RawDataset
from topmost.preprocessing import Preprocessing

from tqdm import tqdm

from . import _plot
from ._fastopic import fastopic
from ._utils import Logger, check_fitted, DocEmbedModel

from typing import List, Tuple, Union, Mapping, Any, Callable, Iterable, Literal


logger = Logger("WARNING")


class FASTopic:
    def __init__(
        self,
        num_topics: int,
        preprocessing: Preprocessing=None,
        doc_embed_model: Union[str, callable]="all-MiniLM-L6-v2",
        num_top_words: int=15,
        DT_alpha: float=3.0,
        TW_alpha: float=2.0,
        theta_temp: float=1.0,
        epochs: int=200,
        learning_rate: float=0.002,
        device: str=None,
        normalize_embeddings: bool=False,
        save_memory: bool=False,
        batch_size: int=None,
        log_interval: int=10,
        verbose: bool=False,
    ):
        """FASTopic initialization.

        Args:
            num_topics: The number of topics.
            preprocessing: preprocessing class from topmost.preprocessing.Preprocessing
            doc_embed_model: The used document embedding model.
                             This can be your callable model that implements `.encode(docs)`.
                             This can also be a model name in sentence-transformers.
                             The default is "all-MiniLM-L6-v2".
            num_top_words: The number of top words to be returned in topics.
            DT_alpha: The sinkhorn alpha between document embeddings and topic embeddings.
            TW_alpha: The sinkhorn alpha between topic embeddings and word embeddings.
            theta_temp: The temperature parameter of the softmax used
                        to compute doc topic distributions during testing.
            epochs: The number of epochs.
            learning_rate: The learning rate.
            device: The device.
            normalize_embeddings: Set this to True to normalize document embeddings.
                                  This parameter may not be effective
                                  when you use your own document embedding model.
            save_memory: Set this to True to learn with a batch of documents at each time.
                         This uses less memory.
            batch_size: The batch size used when save_memory is True.
            log_interval: The interval to print logs during training.
            verbose: Changes the verbosity of the model, Set to True if you want
                     to track the stages of the model.
        """

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.save_memory = save_memory
        self.batch_size = batch_size

        self.num_top_words = num_top_words
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.preprocessing = preprocessing
        self.doc_embed_model = doc_embed_model
        self.vocab = None
        self.doc_embedder = None
        self.train_doc_embeddings = None
        self.normalize_embeddings = normalize_embeddings

        self.beta = None
        self.train_theta = None
        self.model = fastopic(num_topics, theta_temp, DT_alpha, TW_alpha)

        self.log_interval = log_interval
        self.verbose = verbose
        if verbose:
            logger.set_level("DEBUG")
        else:
            logger.set_level("WARNING")

        logger.info(f'use device: {device}')

    def make_optimizer(self, learning_rate: float):
        args_dict = {
            'params': self.model.parameters(),
            'lr': learning_rate,
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def fit(
        self,
        docs: List[str]
    ):
        self.fit_transform(docs)
        return self

    def fit_transform(self, docs: List[str]):
        # Preprocess docs
        data_size = len(docs)
        if self.save_memory:
            assert self.batch_size is not None
        else:
            self.batch_size = data_size

        dataset_device = 'cpu' if self.save_memory else self.device

        self.doc_embedder = DocEmbedModel(self.doc_embed_model, self.normalize_embeddings, self.device)

        dataset = RawDataset(
            docs,
            self.preprocessing,
            batch_size=self.batch_size,
            device=dataset_device,
            pretrained_WE=False,
            contextual_embed=True,
            doc_embed_model=self.doc_embedder,
            embed_model_device=self.device,
            verbose=self.verbose,
        )

        self.train_doc_embeddings = torch.as_tensor(dataset.train_contextual_embed)

        if not self.save_memory:
            self.train_doc_embeddings = self.train_doc_embeddings.to(self.device)

        self.vocab = dataset.vocab
        embed_size = dataset.contextual_embed_size
        vocab_size = dataset.vocab_size

        self.model.init(vocab_size, embed_size)
        self.model = self.model.to(self.device)

        optimizer = self.make_optimizer(self.learning_rate)

        self.model.train()
        for epoch in tqdm(range(1, self.epochs + 1), desc="Training FASTopic"):

            loss_rst_dict = defaultdict(float)

            for batch_data in dataset.train_dataloader:

                batch_bow = batch_data[:, :vocab_size]
                batch_doc_emb = batch_data[:, vocab_size:]

                if self.save_memory:
                    batch_doc_emb = batch_doc_emb.to(self.device)
                    batch_bow = batch_bow.to(self.device)

                rst_dict = self.model(batch_bow, batch_doc_emb)
                batch_loss = rst_dict['loss']

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                for key in rst_dict:
                    loss_rst_dict[key] += rst_dict[key] * batch_data.shape[0]

            if epoch % self.log_interval == 0:
                output_log = f'Epoch: {epoch:03d}'
                for key in loss_rst_dict:
                    output_log += f' {key}: {loss_rst_dict[key] / data_size :.3f}'
                logger.info(output_log)

        self.beta = self.get_beta()
        self.top_words = self.get_top_words(self.num_top_words)
        self.train_theta = self.transform(self, doc_embeddings=self.train_doc_embeddings)

        return self.top_words, self.train_theta

    def transform(
            self,
            docs: List[str]=None,
            doc_embeddings: np.ndarray=None
        ):

        if docs is None and doc_embeddings is None:
            raise ValueError("Must set either docs or doc_embeddings.")

        if doc_embeddings is None and self.doc_embedder is None:
            raise ValueError("Must set doc embeddings.")

        if doc_embeddings is None:
            doc_embeddings = torch.as_tensor(self.doc_embedder.encode(docs))
            if not self.save_memory:
                doc_embeddings = doc_embeddings.to(self.device)

        with torch.no_grad():
            self.model.eval()
            theta = self.model.get_theta(doc_embeddings, self.train_doc_embeddings)
            theta = theta.detach().cpu().numpy()

        return theta

    def get_beta(self):
        """
            return beta: topic-word distributions matrix, $K \times V$
        """
        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

    def get_top_words(self, num_top_words=15, verbose=None):
        if verbose is None:
            verbose = self.verbose
        beta = self.get_beta()
        top_words = get_top_words(beta, self.vocab, num_top_words, verbose)
        return top_words

    @property
    def topic_embeddings(self):
        """
            return topic embeddings $K \times L$
        """
        return self.model.topic_embeddings.detach().cpu().numpy()

    @property
    def word_embeddings(self):
        """
            return word embeddings $V \times L$
        """
        return self.model.word_embeddings.detach().cpu().numpy()

    @property
    def transp_DT(self):
        """
            return transp_DT $N \times K$
        """
        return self.model.get_transp_DT(self.train_doc_embeddings)

    def save(
        self,
        path: str
    ):
        """Saves the FASTopic model and its PyTorch model weights to the specified path, like `./fastopic.zip`.

        This method saves the dict attributes of the FASTopic object (`self`) except for `doc_embedder` for lower size.

        Args:
            path (str): The path to save the model files. If the directory doesn't exist, it will be created.

        Returns:
            None
        """
        check_fitted(self)

        path = Path(path)
        parent_dir = path.parent
        if not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)

        instance_dict = {}
        for key, value in self.__dict__.items():
            if key not in ['doc_embedder']:
                instance_dict[key] = value

        state = {
            "instance_dict": instance_dict
        }
        torch.save(state, path)


    @classmethod
    def from_pretrained(
            cls,
            path: str,
            device: str=None
        ):
        """Loads a pre-trained FASTopic model from a saved file.

        This method loads a previously saved FASTopic model instance, and rebuilds the `doc_embedder`.

        Args:
            path: The path to the directory containing the serialized FASTopic object `fastopic.zip`.
            device: Move the loaded model to the device.
                    Make sure that the device is the same
                    if FASTopic was saved with your own document embedding model.
                    For instance, you must set `device="cuda"` if your own document embedding models was on GPU.
                    The device can be others if you use models in sentence-transformers.
        Returns:
            FASTopic: An instance of the FASTopic class loaded from the provided file.

        Raises:
            FileNotFoundError: If the specified `path` does not exist.
        """

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        state = torch.load(path, map_location=device)
        instance_dict = state["instance_dict"]
        instance_dict["device"] = device

        instance = cls.__new__(cls)
        instance.__dict__.update(instance_dict)

        instance.doc_embedder = DocEmbedModel(
            instance_dict["doc_embed_model"],
            instance_dict["normalize_embeddings"],
            instance_dict["device"],
        )

        return instance

    def get_topic(
            self,
            topic_idx: int,
            num_top_words: int=5
        ):

        check_fitted(self)
        words = self.top_words[topic_idx].split()[:num_top_words]
        scores = np.sort(self.beta[topic_idx])[:-(num_top_words + 1):-1]

        return tuple(zip(words, scores))

    def get_topic_weights(self):
        check_fitted(self)

        topic_weights = self.transp_DT.sum(0)
        return topic_weights

    def visualize_topic(self, **args):
        check_fitted(self)
        return _plot.visualize_topic(self, **args)

    def visualize_topic_hierarchy(self, **args):
        check_fitted(self)
        return _plot.visualize_hierarchy(self, **args)

    def topic_activity_over_time(self,
                                 time_slices: List[int],
                                ):
        check_fitted(self)

        topic_activity = self.transp_DT
        topic_activity *= self.transp_DT.shape[0]

        assert len(time_slices) == topic_activity.shape[0]

        df = pd.DataFrame(topic_activity)
        df['time_slices'] = time_slices
        topic_activity = df.groupby('time_slices').mean().to_numpy().transpose()

        return topic_activity

    def visualize_topic_activity(self, **args):
        check_fitted(self)
        return _plot.visualize_activity(self, **args)

    def visualize_topic_weights(self, **args):
        check_fitted(self)
        return _plot.visualize_topic_weights(self, **args)
