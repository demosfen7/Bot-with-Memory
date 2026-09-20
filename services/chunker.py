"""Разбиение текста на чанки с перекрытием (overlap)."""
from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 75

_SPACES_RE = re.compile(r"[^\S\n]+")  # любые пробельные символы, кроме перевода строки
_SPACE_AROUND_NEWLINE_RE = re.compile(r" ?\n ?")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_SENTENCE_END_RE = re.compile(r"[.!?…][\"'»)\]]*\s")


@dataclass(frozen=True)
class Chunk:
    text: str
    chunk_index: int
    start_char: int  # позиции считаются в тексте после normalize_text()
    end_char: int


def normalize_text(text: str) -> str:
    """Схлопывает повторные пробелы и пустые строки, не меняя смысл текста."""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACES_RE.sub(" ", text)
    text = _SPACE_AROUND_NEWLINE_RE.sub("\n", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def _find_break(text: str, hard_end: int, min_end: int) -> int:
    """Ищет лучшую границу чанка в окне [min_end, hard_end].

    Приоритет: абзац -> конец предложения -> перевод строки -> пробел -> жёсткий разрез.
    """
    if min_end >= hard_end:
        return hard_end

    idx = text.rfind("\n\n", min_end, hard_end)
    if idx != -1:
        return idx + 2

    sentence_ends = list(_SENTENCE_END_RE.finditer(text, min_end, hard_end))
    if sentence_ends:
        return sentence_ends[-1].end()

    idx = text.rfind("\n", min_end, hard_end)
    if idx != -1:
        return idx + 1

    idx = text.rfind(" ", min_end, hard_end)
    if idx != -1:
        return idx + 1

    return hard_end


def _snap_to_word_start(text: str, start: int, limit: int) -> int:
    """Если start попал внутрь слова, сдвигает его к началу следующего слова."""
    if start <= 0 or start >= limit:
        return start
    if text[start - 1].isspace() or text[start].isspace():
        return start
    for i in range(start, limit):
        if text[i].isspace():
            return i + 1
    return start  # слово длиннее окна: остаётся жёсткий разрез


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[Chunk]:
    """Делит текст на чанки не длиннее chunk_size символов с перекрытием overlap.

    Пустой текст даёт пустой список. Некорректный overlap (отрицательный или
    не меньше chunk_size) исправляется, поэтому цикл всегда завершается.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size должен быть положительным числом")
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = chunk_size // 2

    normalized = normalize_text(text)
    total = len(normalized)
    chunks: list[Chunk] = []
    start = 0

    while start < total:
        hard_end = min(start + chunk_size, total)
        if hard_end == total:
            end = total
        else:
            # Не режем слишком рано: чанк должен быть длиннее overlap, иначе не будет прогресса
            min_end = start + max(chunk_size // 2, overlap + 1)
            end = _find_break(normalized, hard_end, min_end)

        # Обрезаем пробелы по краям, сохраняя корректные позиции в тексте
        raw = normalized[start:end]
        piece_start = start + (len(raw) - len(raw.lstrip()))
        piece_end = start + len(raw.rstrip())
        if piece_end > piece_start:
            chunks.append(Chunk(normalized[piece_start:piece_end], len(chunks), piece_start, piece_end))

        if end >= total:
            break
        # Следующий чанк начинается с перекрытием, но всегда правее предыдущего старта
        start = max(end - overlap, start + 1)
        start = _snap_to_word_start(normalized, start, end)

    return chunks
