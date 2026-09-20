"""Конфигурация бота: чтение .env и проверка значений."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Корень проекта: пути из .env считаются относительно него, а не текущей папки запуска
BASE_DIR = Path(__file__).resolve().parent

# Сколько последних реплик оставляем после резюмирования истории
SUMMARY_KEEP_MESSAGES = 6

# Значения-заглушки из .env.example не считаются настоящими ключами
_PLACEHOLDERS = {"your_telegram_bot_token", "your_openai_api_key"}


class ConfigError(Exception):
    """Ошибка конфигурации; текст безопасно показывать в консоли."""


@dataclass(frozen=True)
class Settings:
    # Секреты не попадают в repr(), чтобы случайно не оказаться в логах
    bot_token: str = field(repr=False)
    openai_api_key: str = field(repr=False)

    chat_model: str
    embed_model: str

    chroma_path: Path
    uploads_path: Path
    db_path: Path

    short_memory_messages: int
    short_memory_summary_after: int

    chunk_size: int
    chunk_overlap: int
    rag_top_k: int
    rag_max_distance: float
    max_document_size_mb: int
    max_answer_tokens: int

    log_level: str

    @property
    def max_document_bytes(self) -> int:
        return self.max_document_size_mb * 1024 * 1024


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value or value in _PLACEHOLDERS:
        raise ConfigError(
            f"Не задана переменная окружения {name}. "
            "Скопируйте .env.example в .env и впишите настоящее значение."
        )
    return value


def _get_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"Переменная {name} должна быть целым числом, сейчас: {raw!r}.") from None
    if value < minimum:
        raise ConfigError(f"Переменная {name} должна быть не меньше {minimum}, сейчас: {value}.")
    return value


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ConfigError(f"Переменная {name} должна быть числом, сейчас: {raw!r}.") from None


def _get_path(name: str, default: str) -> Path:
    path = Path(os.getenv(name, "").strip() or default).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def load_settings() -> Settings:
    """Читает .env и возвращает проверенные настройки."""
    load_dotenv(BASE_DIR / ".env")

    # Обязательные ключи проверяем по отдельности, чтобы сообщение было точным
    bot_token = _required("BOT_TOKEN")
    openai_api_key = _required("OPENAI_API_KEY")

    chunk_size = _get_int("CHUNK_SIZE", 500, minimum=50)
    chunk_overlap = _get_int("CHUNK_OVERLAP", 75, minimum=0)
    if chunk_overlap >= chunk_size:
        raise ConfigError("CHUNK_OVERLAP должен быть меньше CHUNK_SIZE.")

    summary_after = _get_int("SHORT_MEMORY_SUMMARY_AFTER", 20, minimum=0)
    if 0 < summary_after <= SUMMARY_KEEP_MESSAGES:
        raise ConfigError(
            f"SHORT_MEMORY_SUMMARY_AFTER должен быть больше {SUMMARY_KEEP_MESSAGES} "
            "(или 0, чтобы отключить резюмирование)."
        )

    log_level = (os.getenv("LOG_LEVEL", "").strip() or "INFO").upper()

    return Settings(
        bot_token=bot_token,
        openai_api_key=openai_api_key,
        chat_model=os.getenv("CHAT_MODEL", "").strip() or "gpt-5-mini",
        embed_model=os.getenv("EMBED_MODEL", "").strip() or "text-embedding-3-small",
        chroma_path=_get_path("CHROMA_PATH", "./data/chroma"),
        uploads_path=_get_path("UPLOADS_PATH", "./data/uploads"),
        db_path=(BASE_DIR / "data" / "app.db").resolve(),
        short_memory_messages=_get_int("SHORT_MEMORY_MESSAGES", 10, minimum=2),
        short_memory_summary_after=summary_after,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        rag_top_k=_get_int("RAG_TOP_K", 5, minimum=1),
        rag_max_distance=_get_float("RAG_MAX_DISTANCE", 0.7),
        max_document_size_mb=_get_int("MAX_DOCUMENT_SIZE_MB", 20, minimum=1),
        max_answer_tokens=_get_int("MAX_ANSWER_TOKENS", 4000, minimum=200),
        log_level=log_level,
    )
