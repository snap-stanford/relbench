from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch_frame.config import TextEmbedderConfig
from torch_frame.data.stats import StatType


class UniversalTextEncoder:
    r"""A universal text encoder that uses a pre-trained SentenceTransformer
    to encode text columns into a shared semantic space.

    Args:
        model_name (str): Name of the SentenceTransformer model to use.
            Default: "all-MiniLM-L6-v2" (fast and effective).
        device (torch.device, optional): Device to load the model on.
        batch_size (int): Batch size for encoding.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[torch.device] = None,
        batch_size: int = 256,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=str(self.device))

    def __call__(self, sentences: List[str]) -> Tensor:
        embeddings = self.model.encode(
            sentences,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            device=str(self.device),
            show_progress_bar=False,
        )
        return embeddings

    @property
    def out_channels(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def get_config(self) -> TextEmbedderConfig:
        r"""Returns a TextEmbedderConfig compatible with PyTorch Frame."""
        return TextEmbedderConfig(
            text_embedder=self,
            batch_size=self.batch_size,
        )


class TransferHeteroEncoder(torch.nn.Module):
    r"""A heterogeneous encoder designed for transfer learning.
    
    It assumes that text features are already encoded by UniversalTextEncoder
    and provides a shared embedding space for categorical features.
    
    Args:
        node_to_col_names_dict: Mapping from node type to Stype-wise column names.
        col_stats: Statistics for the columns (needed for initialization).
        channels (int): Output embedding dimension.
    """
    def __init__(
        self,
        node_to_col_names_dict,
        node_to_col_stats,
        channels: int = 128,
        # We can add more args to match HeteroEncoder if needed
    ):
        super().__init__()
        
        from relbench.modeling.nn import HeteroEncoder
        
        # We wrap the standard HeteroEncoder. 
        # Future extensions can implement shared categorical embeddings here.
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict=node_to_col_names_dict,
            node_to_col_stats=node_to_col_stats,
            torch_frame_model_kwargs={
                "channels": channels,
                "num_layers": 2, 
            }
        )
        
    def forward(self, tf_dict):
        return self.encoder(tf_dict)
