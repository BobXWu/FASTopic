import numpy as np
import itertools

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.cluster import hierarchy as sch

import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from typing import Callable, List, Union


def wrap_topic_idx(
        topic_model,
        top_n: int=None,
        topic_idx: List[int]=None
    ):

    topic_weights = topic_model.get_topic_weights()

    if top_n is None and topic_idx is None:
        top_n = 5
        topic_idx = np.argsort(topic_weights)[:-(top_n + 1):-1]
    elif top_n is not None:
        assert (top_n > 0) and (topic_idx is None)
        topic_idx = np.argsort(topic_weights)[:-(top_n + 1):-1]

    return topic_idx


def visualize_topic(topic_model,
                    top_n: int=None,
                    topic_idx: List[int]=None,
                    n_label_words=5,
                    width: int = 250,
                    height: int = 250
                ):

    topic_idx = wrap_topic_idx(topic_model, top_n, topic_idx)

    top_words = topic_model.top_words
    beta = topic_model.get_beta()

    subplot_titles = [f"Topic {i}" for i in topic_idx]

    columns = 4
    rows = int(np.ceil(len(topic_idx) / columns))

    colors = itertools.cycle(["#D55E00", "#0072B2", "#CC79A7", "#E69F00", "#56B4E9", "#009E73", "#F0E442"])

    fig = make_subplots(rows=rows,
                        cols=columns,
                        shared_xaxes=False,
                        horizontal_spacing=.1,
                        subplot_titles=subplot_titles,
                    )

    row = 1
    column = 1
    for i in topic_idx:
        words = top_words[i].split()[:n_label_words][::-1]
        scores = np.sort(beta[i])[:-(n_label_words + 1):-1][::-1]

        fig.add_trace(
                go.Bar(x=scores,
                    y=words,
                    orientation='h',
                    marker_color=next(colors)
                ),
                row=row,
                col=column
            )

        if column == columns:
            column = 1
            row += 1
        else:
            column += 1

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        title={
            'text': f"Topic-Word Distributions",
            'x': .5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width * 4,
        height=height * rows if rows > 1 else height * 1.3,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig


def visualize_activity(topic_model,
                       topic_activity: np.ndarray,
                       time_slices: Union[np.ndarray, List],
                       top_n: int=None,
                       topic_idx: List[int]=None,
                       n_label_words:int=5,
                       title: str="<b>Topics Activity over Time</b>",
                       width: int=1000,
                       height: int=600
                    ):

    topic_idx = wrap_topic_idx(topic_model, top_n, topic_idx)

    colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00", "#0072B2", "#CC79A7"]

    fig = go.Figure()
    topic_top_words = topic_model.top_words

    legends = []
    for i, words in enumerate(topic_top_words):
        legends.append(f"{i}_{'_'.join(words.split()[:n_label_words])}")

    labels = np.unique(time_slices).tolist()

    for i, k in enumerate(topic_idx):

        fig.add_trace(go.Scatter(
            x=labels,
            y=topic_activity[k].tolist(),
            mode='lines',
            marker_color=colors[i % 7],
            hoverinfo="text",
            name=legends[k],
            hovertext=legends[k])
        )

    # Styling of the visualization
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.update_layout(
        yaxis_title="Topic Weight",
        title={
            'text': f"{title}",
            'y': .95,
            'x': 0.40,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        template="simple_white",
        width=width,
        height=height,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        )
    )

    return fig


def visualize_topic_weights(topic_model,
                            top_n: int=50,
                            topic_idx: List[int]=None,
                            n_label_words: int=5,
                            title: str="<b>Topic Weights</b>",
                            width: int=1000,
                            height: int=1000,
                            _sort: bool=True
                        ):

    topic_weights = topic_model.get_topic_weights()
    topic_idx = wrap_topic_idx(topic_model, top_n, topic_idx)

    labels = []
    vals = []
    topic_top_words = topic_model.top_words

    for i in topic_idx:
        words = topic_top_words[i]
        labels.append(f"{i}_{'_'.join(words.split()[:n_label_words])}")
        vals.append(topic_weights[i])

    if _sort:
        sorted_idx = np.argsort(vals)
        labels = np.asarray(labels)[sorted_idx].tolist()
        vals = np.asarray(vals)[sorted_idx].tolist()

    # Create Figure
    fig = go.Figure(go.Bar(
        x=vals,
        y=labels,
        marker=dict(
            color='#C8D2D7',
            line=dict(
                color='#6E8484',
                width=1),
        ),
        orientation='h')
    )

    fig.update_layout(
        xaxis_title="Weight",
        title={
            'text': f"{title}",
            'y': .95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        template="simple_white",
        width=width,
        height=height,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )

    return fig


def visualize_hierarchy(topic_model,
                        orientation: str = "left",
                        width: int = 1000,
                        height: int = 1000,
                        linkage_function: Callable = None,
                        distance_function: Callable = None,
                        n_label_words: int = 5,
                        color_threshold: int = None
                    ):

    topic_embeddings = topic_model.topic_embeddings

    if distance_function is None:
        # distance_function = lambda x: 1 - cosine_similarity(x)
        distance_function = euclidean_distances

    if linkage_function is None:
        linkage_function = lambda x: sch.linkage(x, 'ward', optimal_ordering=True)

    topic_top_words = topic_model.top_words
    labels = []
    for i, words in enumerate(topic_top_words):
        labels.append(f"{i}_{'_'.join(words.split()[:n_label_words])}")

    fig = ff.create_dendrogram(
        topic_embeddings,
        orientation=orientation,
        labels=labels,
        distfun=distance_function,
        linkagefun=linkage_function,
        color_threshold=color_threshold
    )

    fig.update_layout({'width': width, 'height': height})

    return fig
