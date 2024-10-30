from typing import List, Optional

import torch

# Please run `pip install -U sentence-transformers`
from sentence_transformers import SentenceTransformer
from torch import Tensor


class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        return self.model.encode(sentences, convert_to_tensor=True)
