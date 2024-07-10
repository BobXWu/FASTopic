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


def model_test(model, dataset, num_topics):
    docs = dataset.train_texts
    top_words, train_theta = model.fit_transform(docs)
    test_theta = model.transform(dataset.test_texts)

    assert len(top_words) == num_topics
    assert train_theta.shape[0] == len(docs)
    assert test_theta.shape[0] == len(dataset.test_texts)

    model.get_topic(0)
    model.visualize_topic()
    model.visualize_topic_hierarchy()
    model.visualize_topic_weights(top_n=10, height=500)

    time_slices = dataset.train_times
    act = model.topic_activity_over_time(time_slices)
    model.visualize_topic_activity(top_n=10, topic_activity=act, time_slices=time_slices)

    assert act.shape == (num_topics, len(np.unique(time_slices)))

def test_models(cache_path, num_topics):
    download_dataset("NYT", cache_path=f"{cache_path}/datasets")
    dataset = DynamicDataset(f"{cache_path}/datasets/NYT", as_tensor=False)

    model = FASTopic(num_topics=num_topics, epochs=1, verbose=True)
    model_test(model, dataset, num_topics)

    model = FASTopic(num_topics=num_topics, epochs=1, verbose=True, save_memory=True, batch_size=len(dataset.train_texts) // 2)
    model_test(model, dataset, num_topics)
