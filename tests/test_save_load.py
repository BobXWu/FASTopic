import numpy as np
import pytest

import sys
sys.path.append('../')

from fastopic import FASTopic
from topmost.data import download_dataset, DynamicDataset


@pytest.fixture
def cache_path():
    return './pytest_cache/'


@pytest.fixture
def num_topics():
    return 10


def test(cache_path, num_topics):
    download_dataset('NYT', cache_path='./datasets')
    dataset = DynamicDataset("./datasets/NYT", as_tensor=False)

    model = FASTopic(num_topics=num_topics, epochs=1)
    docs = dataset.train_texts
    model.fit_transform(docs)

    path = f"{cache_path}/model.zip"
    model.save_model(path)

    new_model = FASTopic(num_topics=num_topics)
    new_model.load_model(path)
