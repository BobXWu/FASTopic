# FASTopic

![stars](https://img.shields.io/github/stars/bobxwu/FASTopic?logo=github)
[![PyPI](https://img.shields.io/pypi/v/fastopic)](https://pypi.org/project/topmost)
[![Downloads](https://static.pepy.tech/badge/fastopic)](https://pepy.tech/project/fastopic)
[![LICENSE](https://img.shields.io/github/license/bobxwu/fastopic)](https://www.apache.org/licenses/LICENSE-2.0/)
[![arXiv](https://img.shields.io/badge/arXiv-2405.17978-<COLOR>.svg)](https://arxiv.org/pdf/2405.17978.pdf)
[![Contributors](https://img.shields.io/github/contributors/bobxwu/fastopic)](https://github.com/bobxwu/fastopic/graphs/contributors/)


FASTopic is a fast, adaptive, stable, and transferable topic modeling package.
It leverages pretrained Transformers to produce document embeddings, and discovers latent topics through the optimal transport between document, topic, and word embeddings.
This brings about a neat and efficient topic modeling paradigm, different from traditional probabilistic, VAE-based, and clustering-based models.


<img src='docs/img/illustration.svg' with='300pt'></img>


## Installation

Install FASTopic with `pip`:

```bash
pip install fastopic
```

Otherwise, install FASTopic from the source:

```bash
git clone https://github.com/bobxwu/FASTopic.git
cd FASTopic && python setup.py install
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

### 1. Try FASTopic on your dataset


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


### 2. Topic activity over time

After training, we can compute the activity of each topic at each time slice.

```python
topic_activity = model.topic_activity_over_time(time_slices)
```


## Citation

If you want to use our package, please cite as

    @article{wu2024fastopic,
        title={FASTopic: A Fast, Adaptive, Stable, and Transferable Topic Modeling Paradigm},
        author={Wu, Xiaobao and Nguyen, Thong and Zhang, Delvin Ce and Wang, William Yang and Luu, Anh Tuan},
        journal={arXiv preprint arXiv:2405.17978},
        year={2024}
    }
