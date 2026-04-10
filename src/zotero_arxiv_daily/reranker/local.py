import logging
import warnings

import numpy as np
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import BaseReranker, register_reranker


SAFE_FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@register_reranker("local")
class LocalReranker(BaseReranker):
    def _is_remote_code_load_error(self, exc: Exception) -> bool:
        message = f"{type(exc).__name__}: {exc}"
        return any(
            marker in message
            for marker in [
                "multiple values for keyword argument 'trust_remote_code'",
                "No module named 'custom_st'",
            ]
        )

    def _is_huggingface_availability_error(self, exc: Exception) -> bool:
        message = f"{type(exc).__name__}: {exc}"
        return any(
            marker in message
            for marker in [
                "503 Service Unavailable",
                "couldn't connect to 'https://huggingface.co'",
                "LocalEntryNotFoundError",
                "ReadTimeout",
                "ConnectError",
            ]
        )

    def _lexical_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        texts = [(text or "").strip() for text in [*s1, *s2]]
        if not any(texts):
            return np.zeros((len(s1), len(s2)))

        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            features = vectorizer.fit_transform(texts)
        except ValueError:
            return np.zeros((len(s1), len(s2)))

        left = features[: len(s1)]
        right = features[len(s1):]
        return (left @ right.T).toarray()

    def _load_encoder(self):
        from sentence_transformers import SentenceTransformer

        primary_model = self.config.reranker.local.model
        fallback_model = self.config.reranker.local.get("fallback_model", SAFE_FALLBACK_MODEL)

        logger.info(f"Loading local reranker model: {primary_model}")
        try:
            return SentenceTransformer(primary_model, trust_remote_code=True)
        except Exception as exc:
            if not self._is_remote_code_load_error(exc) or not fallback_model:
                raise
            logger.warning(
                f"Primary reranker model '{primary_model}' failed to load due to remote-code incompatibility: {exc}. "
                f"Falling back to '{fallback_model}'."
            )
            return SentenceTransformer(fallback_model)

    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        if not self.config.executor.debug:
            from transformers.utils import logging as transformers_logging
            from huggingface_hub.utils import logging as hf_logging
    
            transformers_logging.set_verbosity_error()
            hf_logging.set_verbosity_error()
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", category=FutureWarning)

        try:
            encoder = self._load_encoder()
        except Exception as exc:
            if not (
                self._is_remote_code_load_error(exc)
                or self._is_huggingface_availability_error(exc)
            ):
                raise
            logger.warning(
                f"Embedding reranker unavailable ({exc}). Falling back to local TF-IDF lexical similarity."
            )
            return self._lexical_similarity_score(s1, s2)
        if self.config.reranker.local.encode_kwargs:
            encode_kwargs = self.config.reranker.local.encode_kwargs
        else:
            encode_kwargs = {}
        logger.info(f"Encoding {len(s1)} candidate abstracts...")
        s1_feature = encoder.encode(s1, **encode_kwargs, show_progress_bar=True)
        logger.info(f"Encoding {len(s2)} Zotero abstracts...")
        s2_feature = encoder.encode(s2, **encode_kwargs, show_progress_bar=True)
        logger.info("Computing similarity matrix...")
        sim = encoder.similarity(s1_feature, s2_feature)
        return sim.numpy()
