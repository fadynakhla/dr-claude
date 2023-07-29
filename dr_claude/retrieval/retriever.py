from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModel
from langchain.vectorstores import FAISS

from dr_claude.retrieval.embeddings import HuggingFaceEncoderEmbeddings, HuggingFaceEncoderEmbeddingsConfig


class HuggingFAISS(FAISS):

    @classmethod
    def from_model_config_and_texts(
        cls,
        texts: List[str],
        model_config: HuggingFaceEncoderEmbeddingsConfig,
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        embeddings = HuggingFaceEncoderEmbeddings.from_config(model_config)
        return cls.from_texts(texts, embeddings, metadatas, ids)


if __name__ == "__main__":
    model_config = HuggingFaceEncoderEmbeddingsConfig(
        model_name_or_path="bert-base-uncased",
        device="cpu",
        pooling="cls",
    )
    texts = ["This is a test", "This is another test", "foo bar baz"]
    faiss = HuggingFAISS.from_model_config_and_texts(texts, model_config)
    print(faiss.similarity_search_with_score("This is a test"))
