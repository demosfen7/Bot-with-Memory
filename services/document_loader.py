"""Безопасные имена файлов и извлечение текста из TXT, PDF и DOCX."""
from __future__ import annotations

import logging
import os
import re
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".txt", ".pdf", ".docx")

# Оставляем буквы (в том числе кириллицу), цифры, точку, дефис, пробел, скобки и "_"
_UNSAFE_CHARS_RE = re.compile(r"[^\w.\- ()]+", re.UNICODE)


class DocumentLoadError(Exception):
    """Ошибка чтения документа; сообщение безопасно показывать пользователю."""


def sanitize_filename(filename: str, max_length: int = 100) -> str:
    """Делает из имени файла безопасное: без путей, спецсимволов и слишком длинных имён."""
    name = unicodedata.normalize("NFC", filename or "")
    # Отбрасываем любые каталоги (и Windows-, и Unix-стиля) — защита от path traversal
    name = name.replace("\\", "/").split("/")[-1]
    name = "".join(ch for ch in name if ch.isprintable())
    name = _UNSAFE_CHARS_RE.sub("_", name).strip(" .")

    stem, ext = os.path.splitext(name)
    ext = ext.lower()
    stem = stem[: max(1, max_length - len(ext))].strip(" .") or "document"
    return stem + ext


def extract_text(path: Path) -> str:
    """Извлекает текст из файла по его расширению. Блокирующая функция."""
    ext = path.suffix.lower()
    if ext == ".txt":
        text = _read_txt(path)
    elif ext == ".pdf":
        text = _read_pdf(path)
    elif ext == ".docx":
        text = _read_docx(path)
    else:
        raise DocumentLoadError("Формат не поддерживается. Отправьте файл PDF, DOCX или TXT.")

    text = text.replace("\x00", "")
    if not text.strip():
        if ext == ".pdf":
            raise DocumentLoadError(
                "В этом PDF не удалось найти текст. Возможно, это скан или картинка. "
                "Загрузите PDF с текстовым слоем."
            )
        raise DocumentLoadError("Документ пустой: в нём нет текста.")
    return text


def _read_txt(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        logger.error("Не удалось прочитать TXT: %s", exc)
        raise DocumentLoadError("Не удалось прочитать файл.") from exc

    # Пробуем популярные кодировки; cp1251 нужна для старых русскоязычных файлов
    for encoding in ("utf-8-sig", "utf-16", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader  # импорт внутри функции: не нужен, пока не загружают PDF

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise DocumentLoadError("PDF защищён паролем. Загрузите файл без пароля.")

        pages: list[str] = []
        for number, page in enumerate(reader.pages, start=1):
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                # Одна «сломанная» страница не должна ронять весь документ
                logger.warning("Не удалось извлечь текст со страницы %d PDF", number, exc_info=True)
        return "\n\n".join(pages)
    except DocumentLoadError:
        raise
    except Exception as exc:
        logger.error("Ошибка чтения PDF", exc_info=True)
        raise DocumentLoadError("Не удалось прочитать PDF. Файл повреждён или имеет неподдерживаемый формат.") from exc


def _read_docx(path: Path) -> str:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document = Document(str(path))
        blocks: list[str] = []
        # Абзацы и таблицы читаем в том порядке, в котором они идут в документе
        for block in document.iter_inner_content():
            if isinstance(block, Paragraph):
                blocks.append(block.text)
            elif isinstance(block, Table):
                for row in block.rows:
                    cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if cells:
                        blocks.append(" | ".join(cells))
        return "\n".join(blocks)
    except Exception as exc:
        logger.error("Ошибка чтения DOCX", exc_info=True)
        raise DocumentLoadError("Не удалось прочитать DOCX. Файл повреждён или имеет неподдерживаемый формат.") from exc
