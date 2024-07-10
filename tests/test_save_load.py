from numpy.testing import assert_almost_equal
import pytest
import shutil

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
    download_dataset('NYT', cache_path=f'{cache_path}/datasets')
    dataset = DynamicDataset("./datasets/NYT", as_tensor=False)

    model = FASTopic(num_topics=num_topics, epochs=1)
    docs = dataset.train_texts
    model.fit_transform(docs)
    beta = model.get_beta()

    path = f"{cache_path}/tmp_save/fastopic.zip"
    model.save(path)

    new_model = FASTopic.from_pretrained(path, device='cuda')
    new_model.transform(dataset.test_texts)
    new_beta = new_model.get_beta()
    assert_almost_equal(beta, new_beta)

    new_model = FASTopic.from_pretrained(path, device='cpu')
    new_model.transform(dataset.test_texts)
    new_beta = new_model.get_beta()
    assert_almost_equal(beta, new_beta)

    shutil.rmtree(f"{cache_path}/tmp_save")
