"""Жизненный цикл документа: загрузка -> извлечение текста -> чанки -> эмбеддинги -> удаление."""
from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

from config import Settings
from services.chunker import chunk_text
from services.document_loader import DocumentLoadError, extract_text, sanitize_filename
from services.embeddings import EmbeddingService
from services.registry import DocumentRecord, DocumentRegistry, utc_now_iso
from services.vector_store import VectorStore

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], Awaitable[None]]


class DocumentService:
    def __init__(
        self,
        settings: Settings,
        registry: DocumentRegistry,
        store: VectorStore,
        embeddings: EmbeddingService,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._store = store
        self._embeddings = embeddings
        self._uploads_root = settings.uploads_path.resolve()

    def _user_dir(self, user_id: int) -> Path:
        return self._uploads_root / str(int(user_id))

    def new_upload(self, user_id: int, original_name: str) -> tuple[str, str, Path]:
        """Готовит doc_id, безопасное имя и путь для сохранения: uploads/{user_id}/{doc_id}_{имя}."""
        doc_id = str(uuid.uuid4())
        safe_name = sanitize_filename(original_name)
        user_dir = self._user_dir(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        path = (user_dir / f"{doc_id}_{safe_name}").resolve()
        if path.parent != user_dir.resolve():  # страховка от path traversal
            raise DocumentLoadError("Недопустимое имя файла.")
        return doc_id, safe_name, path

    async def index_file(
        self,
        user_id: int,
        doc_id: str,
        filename: str,
        path: Path,
        progress: ProgressCallback | None = None,
    ) -> DocumentRecord:
        """Индексирует уже скачанный файл. При любой ошибке откатывает все следы документа."""
        try:
            size = path.stat().st_size
            if size == 0:
                raise DocumentLoadError("Файл пустой.")
            if size > self._settings.max_document_bytes:
                raise DocumentLoadError(
                    f"Файл слишком большой. Максимальный размер — {self._settings.max_document_size_mb} МБ."
                )

            uploaded_at = utc_now_iso()
            self._registry.add_document(doc_id, user_id, filename, str(path), uploaded_at, size)
            logger.info("Документ загружен: user=%s doc=%s имя=%s размер=%d Б", user_id, doc_id, filename, size)

            text = await asyncio.to_thread(extract_text, path)
            chunks = chunk_text(text, self._settings.chunk_size, self._settings.chunk_overlap)
            if not chunks:
                raise DocumentLoadError("Документ пустой: в нём нет текста.")
            logger.info("Текст извлечён (%d симв.), создано чанков: %d (doc=%s)", len(text), len(chunks), doc_id)

            if progress:
                await progress(f"Текст извлечён ({len(text)} симв.). Индексирую документ: чанков — {len(chunks)}…")

            vectors = await self._embeddings.embed_texts([c.text for c in chunks])
            await asyncio.to_thread(
                self._store.add_chunks, user_id, doc_id, filename, uploaded_at, chunks, vectors
            )
            self._registry.mark_ready(doc_id, len(chunks))
            logger.info("Индексация завершена: user=%s doc=%s чанков=%d", user_id, doc_id, len(chunks))
        except BaseException:
            # BaseException — чтобы откатиться и при отмене задачи (CancelledError)
            self._rollback(user_id, doc_id, path)
            raise

        record = self._registry.get_document(user_id, doc_id)
        assert record is not None
        return record

    def _rollback(self, user_id: int, doc_id: str, path: Path) -> None:
        """Убирает файл, чанки и запись реестра неудавшегося документа (без исключений наружу)."""
        for action, func in (
            ("удаление чанков", lambda: self._store.delete_document(user_id, doc_id)),
            ("удаление записи реестра", lambda: self._registry.delete_document(user_id, doc_id)),
            ("удаление файла", lambda: path.unlink(missing_ok=True)),
        ):
            try:
                func()
            except Exception:
                logger.exception("Откат индексации: не удалось выполнить «%s» (doc=%s)", action, doc_id)

    def delete(self, user_id: int, doc_id: str) -> bool:
        """Удаляет документ пользователя целиком. False — если такого документа у него нет."""
        record = self._registry.get_document(user_id, doc_id)
        if record is None:
            return False

        # Сначала вектора: если Chroma упадёт, запись и файл останутся, и удаление можно повторить
        self._store.delete_document(user_id, doc_id)
        self._remove_file(user_id, record.saved_path)
        self._registry.delete_document(user_id, doc_id)
        logger.info("Документ удалён: user=%s doc=%s", user_id, doc_id)
        return True

    def cleanup_unfinished(self) -> int:
        """При запуске убирает документы, индексация которых оборвалась. Возвращает их число."""
        records = self._registry.list_unfinished()
        for record in records:
            self._rollback(record.user_id, record.id, Path(record.saved_path))
        if records:
            logger.warning("Удалены недоиндексированные документы: %d", len(records))
        return len(records)

    def _remove_file(self, user_id: int, saved_path: str) -> None:
        path = Path(saved_path).resolve()
        # Удаляем только внутри папки этого пользователя
        if path.parent != self._user_dir(user_id).resolve():
            logger.error("Путь файла вне папки пользователя, файл не удалён: user=%s", user_id)
            return
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Не удалось удалить файл документа (user=%s)", user_id)
