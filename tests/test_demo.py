import pytest
import shutil
from numpy.testing import assert_almost_equal
from topmost import Preprocess
from sklearn.datasets import fetch_20newsgroups

import sys
sys.path.append("../")

from fastopic import FASTopic

@pytest.fixture
def cache_path():
    return "./pytest_cache/"


@pytest.fixture
def num_topics():
    return 10


def test(cache_path, num_topics):
    docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
    docs = docs[:100]

    model = FASTopic(num_topics)
    top_words, doc_topic_dist = model.fit_transform(docs, epochs=1)

    preprocess = Preprocess(vocab_size=10000, stopwords="English")

    model = FASTopic(num_topics, preprocess)
    top_words, doc_topic_dist = model.fit_transform(docs, epochs=1)
    beta = model.get_beta()

    path = f"{cache_path}/tmp_save/fastopic.zip"
    model.save(path)

    loaded_model = FASTopic.from_pretrained(path)

    loaded_beta = loaded_model.get_beta()
    assert_almost_equal(beta, loaded_beta)

    loaded_doc_topic_dist = loaded_model.transform(docs)
    assert_almost_equal(doc_topic_dist, loaded_doc_topic_dist)

    # Keep training
    loaded_model.fit_transform(docs, epochs=1)

    shutil.rmtree(f"{cache_path}/tmp_save")
