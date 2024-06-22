# FASTopic

![stars](https://img.shields.io/github/stars/bobxwu/FASTopic?logo=github)
[![PyPI](https://img.shields.io/pypi/v/fastopic)](https://pypi.org/project/fastopic)
[![Downloads](https://static.pepy.tech/badge/fastopic)](https://pepy.tech/project/fastopic)
[![LICENSE](https://img.shields.io/github/license/bobxwu/fastopic)](https://www.apache.org/licenses/LICENSE-2.0/)
[![arXiv](https://img.shields.io/badge/arXiv-2405.17978-<COLOR>.svg)](https://arxiv.org/pdf/2405.17978.pdf)
[![Contributors](https://img.shields.io/github/contributors/bobxwu/fastopic)](https://github.com/bobxwu/fastopic/graphs/contributors/)


FASTopic is a fast, adaptive, stable, and transferable topic model, different
from previous conventional (LDA), VAE-based (ProLDA, ETM), or clustering-based (Top2Vec, BERTopic) methods.
It leverages optimal transport between the embeddings from pretrained Transformers to model topics and topic distributions of documents.


<img src='docs/img/illustration.svg' with='300pt'></img>

## Tutorials

We release a complete [tutorial](https://github.com/BobXWu/FASTopic/blob/master/tutorials/tutorials_FASTopic.ipynb) on FASTopic.


## Installation

Install FASTopic with `pip`:

```bash
pip install fastopic
```

Otherwise, install FASTopic from the source:

```bash
pip install git+https://github.com/bobxwu/FASTopic.git
```

## Quick Start

Discover topics from 20newsgroups.

```python
from fastopic import FASTopic
from sklearn.datasets import fetch_20newsgroups
from topmost.preprocessing import Preprocessing

docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

preprocessing = Preprocessing(vocab_size=10000, stopwords='English')

model = FASTopic(num_topics=50, preprocessing)
topic_top_words, doc_topic_dist = model.fit_transform(docs)

```

`topic_top_words` is a list of the top words in discovered topics.
`doc_topic_dist` is the topic distributions of documents (doc-topic distributions),
a numpy array with shape $N \times K$ (number of documents $N$ and number of topics $K$).


## Usage

### Try FASTopic on your dataset

```python
from fastopic import FASTopic
from topmost.preprocessing import Preprocessing

# Prepare your dataset.
your_dataset = [
    'doc 1',
    'doc 2', # ...
]

# Preprocess the dataset. This step tokenizes docs, removes stopwords, and sets max vocabulary size, etc..
# Pass your tokenizer as:
#   preprocessing = Preprocessing(vocab_size=your_vocab_size, tokenizer=your_tokenizer, stopwords=your_stopwords_set)
preprocessing = Preprocessing(stopwords='English')

model = FASTopic(num_topics=50, preprocessing)
topic_top_words, doc_topic_dist = model.fit_transform(docs)
```


### Topic info

We can get the top words and their probabilities of a topic.

```python
model.get_topic(topic_idx=36)

(('impeachment', 0.008047104),
 ('mueller', 0.0075936727),
 ('trump', 0.0066773472),
 ('committee', 0.0057785935),
 ('inquiry', 0.005647915))
```

We can visualize these topic info.

```python
fig = model.visualize_topic(top_n=5)
fig.show()
```

<img src='https://github.com/BobXWu/FASTopic/blob/master/tutorials/topic_info.png?raw=true' with='300pt'></img>


### Topic hierarchy

We use the learned topic embeddings and `scipy.cluster.hierarchy` to build a hierarchy of discovered topics.


```python
fig = model.visualize_topic_hierarchy()
fig.show()
```

<img src='https://github.com/BobXWu/FASTopic/blob/master/tutorials/topic_hierarchy.png?raw=true' with='300pt'></img>



### Topic weights

We plot the weights of topics in the given dataset.

```python
fig = model.visualize_topic_weights(top_n=20, height=500)
fig.show()
```

<img src='https://github.com/BobXWu/FASTopic/blob/master/tutorials/topic_weight.png?raw=true' with='300pt'></img>




### Topic activity over time

Topic activity refers to the weight of a topic at a time slice.
We additionally input the time slices of documents, `time_slices` to compute and plot topic activity over time.



```python
act = model.topic_activity_over_time(time_slices)
fig = model.visualize_topic_activity(top_n=6, topic_activity=act, time_slices=time_slices)
fig.show()
```

<img src='https://github.com/BobXWu/FASTopic/blob/master/tutorials/topic_activity.png?raw=true' with='300pt'></img>



## Citation

If you want to use our package, please cite our [paper](https://arxiv.org/pdf/2405.17978.pdf) as

    @article{wu2024fastopic,
        title={FASTopic: A Fast, Adaptive, Stable, and Transferable Topic Modeling Paradigm},
        author={Wu, Xiaobao and Nguyen, Thong and Zhang, Delvin Ce and Wang, William Yang and Luu, Anh Tuan},
        journal={arXiv preprint arXiv:2405.17978},
        year={2024}
    }
