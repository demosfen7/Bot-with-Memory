"""Получение эмбеддингов через OpenAI Embeddings API."""
from __future__ import annotations

import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Сколько текстов отправляем в API за один запрос
EMBED_BATCH_SIZE = 100
# Страховка от слишком длинного запроса (лимит модели — около 8000 токенов)
MAX_QUERY_CHARS = 6000


class EmbeddingService:
    def __init__(self, client: AsyncOpenAI, model: str, batch_size: int = EMBED_BATCH_SIZE) -> None:
        self._client = client
        self._model = model
        self._batch_size = batch_size

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Возвращает эмбеддинги для списка текстов в том же порядке, обрабатывая их батчами."""
        vectors: list[list[float]] = []
        batches = [texts[i : i + self._batch_size] for i in range(0, len(texts), self._batch_size)]
        for number, batch in enumerate(batches, start=1):
            response = await self._client.embeddings.create(model=self._model, input=batch)
            # API возвращает индексы, но на всякий случай сортируем по ним явно
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)
            logger.info("Эмбеддинги: батч %d/%d готов (%d текстов)", number, len(batches), len(batch))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Эмбеддинг одного поискового запроса."""
        vectors = await self.embed_texts([text.strip()[:MAX_QUERY_CHARS]])
        return vectors[0]
