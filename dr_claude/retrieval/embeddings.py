from typing import Any, Dict, List, Union

import pydantic
import torch
import transformers
from langchain.embeddings import base as embeddings_base


class HuggingFaceEncoderEmbeddings(pydantic.BaseModel, embeddings_base.Embeddings):
    """HuggingFace embedding models not using sentence-transformers.

    To use, you should have the ``transformers`` python package installed.
    """

    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    device: str
    model_name_or_path: str
    cache_folder: str = None
    model_kwargs: Dict[str, Any] = pydantic.Field(default_factory=dict)
    encode_kwargs: Dict[str, Any] = pydantic.Field(default_factory=dict)

    def __init__(self, **kwargs: Any):
        """Initialize the huggingface embedding model."""
        super().__init__(**kwargs)
        self.tokenizer: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_name_or_path, cache_dir=self.cache_folder
        )
        self.model: transformers.PreTrainedModel = transformers.AutoModel.from_pretrained(
            pretrained_model_name_or_path=self.model_name_or_path, **self.model_kwargs
        )
        self.model.to(self.device)

    class Config:
        """Configuration for this pydantic object."""

        extra = pydantic.Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.encode(texts, **self.encode_kwargs)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        embedding = self.encode(text, **self.encode_kwargs)
        return embedding.tolist()

    def encode(self, texts: Union[str, List[str]], pooling: str = "cls") -> torch.Tensor:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        model_input = self.tokenizer(texts, return_tensors="pt", padding=True)
        model_input.to(self.device)
        model_output = self.model(
            input_ids=model_input["input_ids"],
            attention_mask=model_input["attention_mask"],
            return_dict=True,
        )
        if pooling == "cls":
            embeddings = model_output.last_hidden_state[:, 0, :].squeeze()
        elif pooling == "mean":
            embeddings = mean_pooling(model_output.last_hidden_state, model_input["attention_mask"])
        else:
            raise ValueError(f"Pooling method {pooling} not supported.")
        return embeddings


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute mean pooling."""
    attention_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * attention_mask_expanded, 1)
    sum_mask = attention_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    pooled_output = sum_embeddings / sum_mask

    return pooled_output
