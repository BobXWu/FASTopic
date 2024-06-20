import numpy as np
import pandas as pd

import torch
from topmost.utils._utils import get_top_words
from topmost.data import RawDataset
from topmost.preprocessing import Preprocessing

from tqdm import tqdm

from ._utils import Logger, check_fitted
from . import _plot
from ._fastopic import fastopic
from ._embedding_model import DocEmbedModel

from typing import List, Tuple, Union, Mapping, Any, Callable, Iterable


logger = Logger("WARNING")


class FASTopic:
    def __init__(self,
                 num_topics: int,
                 preprocessing: Preprocessing=None,
                 doc_embed_model_name=None,
                 num_top_words: int=15,
                 DT_alpha: float=3.0,
                 TW_alpha: float=2.0,
                 theta_temp: float=1.0,
                 epochs: int=200,
                 learning_rate: float=0.002,
                 device: str=None,
                 verbose: bool=False
                ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.num_top_words = num_top_words
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.preprocessing = preprocessing

        self.beta = None
        self.train_theta = None
        self.model = fastopic(num_topics, DT_alpha, TW_alpha, theta_temp)

        self.doc_embed_model = DocEmbedModel(device, doc_embed_model_name)

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

    def fit(self,
            docs: List[str],
            ):

        self.fit_transform(docs)
        return self

    def fit_transform(self,
                      docs: List[str],
                    ):

        # Preprocess docs
        dataset = RawDataset(docs, self.preprocessing, device=self.device, pretrained_WE=False)
        bow = dataset.train_data
        vocab_size = dataset.vocab_size
        self.vocab = dataset.vocab

        # Compute document embeddings
        doc_embeddings = self.doc_embed_model.encode(docs)

        # Initialize document, topic, and word embeddings
        self.model.init(vocab_size, doc_embeddings)
        self.model = self.model.to(self.device)

        optimizer = self.make_optimizer(self.learning_rate)

        self.model.train()
        for epoch in tqdm(range(1, self.epochs + 1), desc="Training FASTopic"):

            rst_dict = self.model(bow)
            batch_loss = rst_dict['loss']

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            if epoch % 5 == 0:
                output_log = f'Epoch: {epoch:03d}'
                for key in rst_dict:
                    output_log += f' {key}: {rst_dict[key]:.3f}'

                logger.info(output_log)

        self.beta = self.get_beta()
        self.top_words = self.get_top_words(self.num_top_words)
        self.train_theta = self.transform(self, doc_embeddings=doc_embeddings)

        return self.top_words, self.train_theta

    def transform(self,
                  docs: List[str]=None,
                  doc_embeddings: np.ndarray=None
                ):

        if docs is None and doc_embeddings is None:
            raise ValueError("Must set either docs or doc_embeddings.")

        if doc_embeddings is None:
            doc_embeddings = self.doc_embed_model.encode(docs)

        doc_embeddings = torch.from_numpy(doc_embeddings).to(self.device)

        with torch.no_grad():
            self.model.eval()
            theta = self.model.get_theta(doc_embeddings)
            theta = theta.detach().cpu().numpy()

        return theta

    def get_beta(self):
        """
            return beta: topic-word distributions matrix, $K \times V$
        """
        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

    def get_top_words(self, num_top_words=15):
        beta = self.get_beta()
        top_words = get_top_words(beta, self.vocab, num_top_words, self.verbose)
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
    def doc_embeddings(self):
        """
            return document embeddings $N \times L$
        """
        return self.doc.word_embeddings.detach().cpu().numpy()

    def save_model(self, path):
        torch.save(self.model.state_dict(), f"{path}.zip")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(f"{path}.zip"))

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

        transp_DT = self.model.transp_DT
        topic_weights = transp_DT.sum(0)
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

        transp_DT = self.model.transp_DT
        transp_DT = transp_DT * transp_DT.shape[0]

        assert len(time_slices) == transp_DT.shape[0]

        df = pd.DataFrame(transp_DT)
        df['time_slices'] = time_slices
        topic_activity = df.groupby('time_slices').mean().to_numpy().transpose()

        return topic_activity

    def visualize_topic_activity(self, **args):
        check_fitted(self)
        return _plot.visualize_activity(self, **args)

    def visualize_topic_weights(self, **args):
        check_fitted(self)
        return _plot.visualize_topic_weights(self, **args)
