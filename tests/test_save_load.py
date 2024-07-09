from numpy.testing import assert_almost_equal
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
    beta = model.get_beta()

    path = f"{cache_path}/models/"
    model.save(path=path, model_name='test_model')

    new_model = FASTopic().from_pretrained(f"{path}/test_model/fastopic.pkl")
    new_beta = new_model.get_beta()

    assert_almost_equal(beta, new_beta)
