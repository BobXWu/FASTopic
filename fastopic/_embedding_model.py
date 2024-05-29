from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Union, Mapping, Any, Callable, Iterable



class DocEmbedModel:
    def __init__(self,
                 device:str,
                 model_name:str=None
                ):

        if model_name is None:
            model_name = "all-MiniLM-L6-v2"

        self.doc_embed_model = SentenceTransformer(model_name, device=device)

    def encode(self, docs:List[str], verbose:bool=False):
        embeddings = self.doc_embed_model.encode(docs, show_progress_bar=verbose)
        return embeddings
