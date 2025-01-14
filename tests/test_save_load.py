import pytest
import shutil
from numpy.testing import assert_almost_equal
from topmost import Preprocess, download_dataset, DynamicDataset

import sys
sys.path.append('../')

from fastopic import FASTopic


@pytest.fixture
def cache_path():
    return './pytest_cache/'


@pytest.fixture
def num_topics():
    return 10


def test(cache_path, num_topics):
    download_dataset('NYT', cache_path=f'{cache_path}/datasets')
    dataset = DynamicDataset("./datasets/NYT", as_tensor=False)
    docs = dataset.train_texts

    model = FASTopic(num_topics, device='cpu')
    model.fit_transform(docs, epochs=1)
    beta = model.get_beta()

    path = f"{cache_path}/tmp_save/fastopic.zip"
    model.save(path)

    new_preprocess = Preprocess(vocab_size=200)
    new_model = FASTopic.from_pretrained(path, device='cuda', preprocess=new_preprocess)
    new_model.transform(dataset.test_texts)
    new_beta = new_model.get_beta()
    assert_almost_equal(beta, new_beta)

    new_model.fit_transform(docs, epochs=1)

    shutil.rmtree(f"{cache_path}/tmp_save")
