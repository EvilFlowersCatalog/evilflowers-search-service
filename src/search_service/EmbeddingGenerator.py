import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from config.Config import Config

logger = logging.getLogger(__name__)


class EmbeddingGenerator:

    def __init__(self):
        logger.info(f"Loading model: {Config.MODEL_NAME} on {Config.DEVICE}")
        self._model = SentenceTransformer(Config.MODEL_NAME, device=Config.DEVICE)
        logger.info("Model loaded")

    def generate_embeddings(self, texts: list[str], normalize: bool = True, show_progress: bool = False) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=Config.BATCH_SIZE,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            device=Config.DEVICE
        )

    def generate_single_embedding(self, text: str, normalize: bool = True) -> np.ndarray:
        embeddings = self.generate_embeddings([text], normalize=normalize)
        return embeddings[0] if len(embeddings) > 0 else np.array([])