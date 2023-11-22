from typing import List

import torch
# Please run `pip install peft` to install the package
from torch import Tensor
# Please run `pip install transformers` to install the package
from transformers import AutoModel, AutoTokenizer


def mean_pooling(last_hidden_state: Tensor, attention_mask) -> Tensor:
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    )
    embedding = torch.sum(last_hidden_state * input_mask_expanded, dim=1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
    return embedding.unsqueeze(1)


class TextToEmbedding:
    def __init__(
        self,
        model: str = "sentence-transformers/all-distilroberta-v1",
        pooling: str = "mean",
        device: torch.device = torch.device("cpu"),
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(device)
        self.device = device
        self.pooling = pooling

    def __call__(self, sentences: List[str]) -> Tensor:
        inputs = self.tokenizer(
            sentences,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        for key in inputs:
            if isinstance(inputs[key], Tensor):
                inputs[key] = inputs[key].to(self.device)
        out = self.model(**inputs)
        mask = inputs["attention_mask"]
        if self.pooling == "mean":
            return mean_pooling(out.last_hidden_state.detach(), mask).squeeze(1).cpu()
        elif self.pooling == "cls":
            return out.last_hidden_state[:, 0, :].detach().cpu()
        else:
            raise ValueError(f"{self.pooling} is not supported.")
