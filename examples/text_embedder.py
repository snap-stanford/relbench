from typing import List, Optional

import torch

# Please run `pip install -U sentence-transformers`
from sentence_transformers import SentenceTransformer
from torch import Tensor


class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d", device=device
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        return torch.from_numpy(self.model.encode(sentences))


class BERTTextEmbedding:
    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer("bert-base-nli-mean-tokens", device=device)

    def __call__(self, sentences: List[str]) -> Tensor:
        return torch.from_numpy(self.model.encode(sentences))


# helper function for loading text embedding choice
def get_text_embedder(text_embedder: str, device: Optional[torch.device] = None):
    if text_embedder == "glove":
        return GloveTextEmbedding(device=device)
    elif text_embedder == "bert":
        return BERTTextEmbedding(device=device)
    else:
        raise ValueError(f"Unknown text embedder: {text_embedder}")
