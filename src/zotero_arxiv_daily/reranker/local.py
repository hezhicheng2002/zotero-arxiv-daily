from .base import BaseReranker, register_reranker
import logging
import warnings
import numpy as np
from loguru import logger
@register_reranker("local")
class LocalReranker(BaseReranker):
    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer
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

        logger.info(f"Loading local reranker model: {self.config.reranker.local.model}")
        encoder = SentenceTransformer(self.config.reranker.local.model, trust_remote_code=True)
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
