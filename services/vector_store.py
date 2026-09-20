"""Векторная база ChromaDB: одна общая коллекция, изоляция данных по user_id."""
from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from services.chunker import Chunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "long_memory"
# Ограничение на размер одной пачки записи в Chroma
ADD_BATCH_SIZE = 200


class VectorStoreError(Exception):
    """Любая ошибка ChromaDB; подробности лежат в логе."""


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    metadata: dict[str, Any]
    distance: float  # косинусное расстояние: 0 — идентично, чем больше, тем дальше

    @property
    def score(self) -> float:
        """Косинусное сходство (1 - distance): чем выше, тем релевантнее."""
        return 1.0 - self.distance

    @property
    def filename(self) -> str:
        return str(self.metadata.get("original_filename", "документ"))

    @property
    def doc_id(self) -> str:
        return str(self.metadata.get("doc_id", ""))

    @property
    def chunk_number(self) -> int:
        """Человеческий номер фрагмента (нумерация с 1)."""
        return int(self.metadata.get("chunk_index", 0)) + 1


@contextmanager
def _chroma_errors(action: str) -> Iterator[None]:
    """Превращает любые исключения ChromaDB в VectorStoreError с записью в лог."""
    try:
        yield
    except VectorStoreError:
        raise
    except Exception as exc:
        logger.exception("Ошибка ChromaDB при операции: %s", action)
        raise VectorStoreError(f"Ошибка ChromaDB ({action})") from exc


def _where(user_id: int, doc_id: str | None = None) -> dict[str, Any]:
    """Фильтр Chroma. user_id присутствует ВСЕГДА — это и есть изоляция пользователей."""
    if doc_id is None:
        return {"user_id": int(user_id)}
    return {"$and": [{"user_id": int(user_id)}, {"doc_id": doc_id}]}


class VectorStore:
    def __init__(self, path: Path) -> None:
        # Chroma не рассчитана на конкурентные записи из потоков, поэтому все вызовы под замком
        self._lock = threading.RLock()
        path.mkdir(parents=True, exist_ok=True)
        with _chroma_errors("инициализация"):
            self._client = chromadb.PersistentClient(
                path=str(path),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            # Эмбеддинги считаем сами (OpenAI), поэтому встроенная функция Chroma не нужна
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,
            )
        logger.info("ChromaDB готова: %s (чанков в базе: %d)", path, self.total_count())

    def total_count(self) -> int:
        with self._lock, _chroma_errors("подсчёт записей"):
            return self._collection.count()

    def add_chunks(
        self,
        user_id: int,
        doc_id: str,
        filename: str,
        uploaded_at: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Сохраняет чанки вместе с текстом, эмбеддингами и метаданными."""
        if len(chunks) != len(embeddings):
            raise VectorStoreError("Число чанков не совпадает с числом эмбеддингов")

        with self._lock, _chroma_errors("добавление чанков"):
            for offset in range(0, len(chunks), ADD_BATCH_SIZE):
                part = chunks[offset : offset + ADD_BATCH_SIZE]
                self._collection.add(
                    ids=[f"{doc_id}:{c.chunk_index}" for c in part],
                    documents=[c.text for c in part],
                    embeddings=embeddings[offset : offset + ADD_BATCH_SIZE],
                    metadatas=[
                        {
                            "user_id": int(user_id),
                            "doc_id": doc_id,
                            "original_filename": filename,
                            "chunk_index": c.chunk_index,
                            "start_char": c.start_char,
                            "end_char": c.end_char,
                            "uploaded_at": uploaded_at,
                            "source_type": "document",
                        }
                        for c in part
                    ],
                )
        logger.info("Сохранено чанков в ChromaDB: %d (user=%s, doc=%s)", len(chunks), user_id, doc_id)

    def search(
        self,
        user_id: int,
        embedding: list[float],
        n_results: int,
        doc_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Семантический поиск только среди чанков пользователя (и, если задан, документа)."""
        with self._lock, _chroma_errors("поиск"):
            result = self._collection.query(
                query_embeddings=[embedding],
                n_results=max(1, n_results),
                where=_where(user_id, doc_id),
                include=["documents", "metadatas", "distances"],
            )

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            RetrievedChunk(text=text, metadata=dict(meta or {}), distance=float(distance))
            for text, meta, distance in zip(documents, metadatas, distances)
        ]

    def count_chunks(self, user_id: int, doc_id: str | None = None) -> int:
        """Сколько чанков есть у пользователя (или у одного его документа)."""
        with self._lock, _chroma_errors("подсчёт чанков"):
            found = self._collection.get(where=_where(user_id, doc_id), include=[])
        return len(found.get("ids") or [])

    def delete_document(self, user_id: int, doc_id: str) -> None:
        """Удаляет все чанки документа. Фильтр по user_id не даёт удалить чужое."""
        with self._lock, _chroma_errors("удаление документа"):
            self._collection.delete(where=_where(user_id, doc_id))
        logger.info("Чанки документа удалены из ChromaDB (user=%s, doc=%s)", user_id, doc_id)
