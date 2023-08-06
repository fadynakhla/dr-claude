from typing import Any, Dict, List, Optional, Union

import pydantic
import torch
import transformers
from langchain.embeddings import base as embeddings_base


class HuggingFaceEncoderEmbeddingsConfig(pydantic.BaseModel):
    model_name_or_path: str
    device: str
    pooling: Optional[str] = None


class HuggingFaceEncoderEmbeddings(embeddings_base.Embeddings):
    """HuggingFace embedding models not using sentence-transformers.

    To use, you should have the ``transformers`` python package installed.
    """

    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    model_name_or_path: str
    device: str
    pooling: str

    def __init__(
        self,
        model_name_or_path: str,
        device: str,
        pooling: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the huggingface embedding model."""
        super().__init__(**kwargs)
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.pooling = pooling if pooling is not None else "cls"
        self.tokenizer: transformers.PreTrainedTokenizer = (
            transformers.AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.model_name_or_path
            )
        )
        self.model: transformers.PreTrainedModel = (
            transformers.AutoModel.from_pretrained(
                pretrained_model_name_or_path=self.model_name_or_path
            )
        )
        self.model.to(self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.encode(texts, self.pooling)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        embedding = self.encode(text, self.pooling)
        return embedding.tolist()

    def encode(self, texts: Union[str, List[str]], pooling: str) -> torch.Tensor:
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
            embeddings = mean_pooling(
                model_output.last_hidden_state, model_input["attention_mask"]
            )
        else:
            raise ValueError(f"Pooling method {pooling} not supported.")
        return embeddings

    @classmethod
    def from_config(
        cls, config: HuggingFaceEncoderEmbeddingsConfig
    ) -> "HuggingFaceEncoderEmbeddings":
        return cls(**config.dict())


def mean_pooling(
    token_embeddings: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Compute mean pooling."""
    attention_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * attention_mask_expanded, 1)
    sum_mask = attention_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    pooled_output = sum_embeddings / sum_mask

    return pooled_output


if __name__ == "__main__":
    embedder = HuggingFaceEncoderEmbeddings(
        model_name_or_path="bert-base-uncased", device="mps"
    )
    print(embedder.embed_documents(["Hello world!", "blah"]))
