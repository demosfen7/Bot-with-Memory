"""Telegram-бот с кратковременной памятью (история диалога) и долговременной (RAG по документам)."""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import openai
from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatType
from aiogram.exceptions import (
    TelegramAPIError,
    TelegramNetworkError,
    TelegramUnauthorizedError,
)
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import BotCommand, CallbackQuery, ErrorEvent, Message
from aiogram.utils.chat_action import ChatActionSender
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.utils.token import TokenValidationError
from openai import AsyncOpenAI

from config import SUMMARY_KEEP_MESSAGES, ConfigError, Settings, load_settings
from services.document_loader import SUPPORTED_EXTENSIONS, DocumentLoadError, sanitize_filename
from services.documents import DocumentService
from services.embeddings import EmbeddingService
from services.rag import ChatService, ChatServiceError, RagService
from services.registry import DocumentRecord, DocumentRegistry
from services.short_memory import ShortMemory
from services.vector_store import VectorStore, VectorStoreError

logger = logging.getLogger("bot")

# Лимит Telegram — 4096 символов; берём с запасом
TELEGRAM_MESSAGE_LIMIT = 4000

START_TEXT = (
    "Привет! Я ассистент с двумя видами памяти.\n\n"
    "Кратковременная память: я помню наш текущий диалог, поэтому можно задавать уточняющие вопросы "
    "(«А сколько это занимает?»).\n\n"
    "Долговременная память: пришлите мне документ PDF, DOCX или TXT — я его прочитаю и запомню. "
    "После загрузки можно задавать вопросы по документу, а я отвечу на основе его содержимого "
    "и покажу источники.\n\n"
    "Команды:\n"
    "/documents — список ваших документов\n"
    "/use_document <номер> — искать только в одном документе\n"
    "/all_documents — снова искать по всем документам\n"
    "/delete_document <номер> — удалить документ\n"
    "/clear_chat — очистить историю диалога\n"
    "/help — подробная справка"
)

HELP_TEXT = (
    "Команды:\n"
    "/start — приветствие\n"
    "/help — эта справка\n"
    "/documents — список загруженных документов (номер, файл, дата, число чанков, id)\n"
    "/use_document <номер> — сделать документ активным: поиск идёт только по нему\n"
    "/all_documents — отменить выбор документа и искать по всем\n"
    "/delete_document <номер> — удалить документ (потребуется подтверждение)\n"
    "/clear_chat — очистить историю диалога; документы останутся\n\n"
    "Как работает память:\n"
    "- Кратковременная: последние реплики нашего диалога и краткое резюме более старой части. "
    "Она хранится в оперативной памяти и очищается при перезапуске бота или по /clear_chat.\n"
    "- Долговременная: ваши документы, разбитые на фрагменты и сохранённые в векторной базе. "
    "Они не пропадают после перезапуска и доступны только вам.\n\n"
    "Чтобы добавить документ, просто отправьте файл PDF, DOCX или TXT."
)

router = Router()
# Бот работает только в личных чатах: данные привязаны к пользователю
router.message.filter(F.chat.type == ChatType.PRIVATE)


class UserLocks:
    """Один замок на пользователя: его сообщения обрабатываются строго по очереди."""

    def __init__(self) -> None:
        self._locks: dict[int, asyncio.Lock] = {}

    def get(self, user_id: int) -> asyncio.Lock:
        return self._locks.setdefault(user_id, asyncio.Lock())


@dataclass
class App:
    """Все сервисы бота; передаётся в обработчики через workflow_data aiogram."""

    settings: Settings
    registry: DocumentRegistry
    documents: DocumentService
    chat: ChatService
    locks: UserLocks = field(default_factory=UserLocks)


# ---------- вспомогательные функции ----------


class RedactingFormatter(logging.Formatter):
    """Форматтер логов, который заменяет секреты (токен бота, ключ OpenAI) на ***."""

    def __init__(self, fmt: str, secrets: list[str]) -> None:
        super().__init__(fmt)
        self._secrets = [s for s in secrets if s]

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        for secret in self._secrets:
            text = text.replace(secret, "***")
        return text


def setup_logging(level: str = "INFO", secrets: list[str] | None = None) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        RedactingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s", secrets or [])
    )
    logging.basicConfig(level=getattr(logging, level, logging.INFO), handlers=[handler], force=True)
    # Библиотеки болтливы на INFO: оставляем только предупреждения
    for noisy in ("httpx", "httpcore", "chromadb", "aiogram.event"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def split_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Режет длинный текст на части не длиннее limit, предпочитая границы абзацев."""
    parts: list[str] = []
    text = text.strip()
    while len(text) > limit:
        cut = text.rfind("\n\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        parts.append(text)
    return parts


async def send_long(message: Message, text: str) -> None:
    for part in split_message(text) or ["(пустой ответ)"]:
        await message.answer(part)


def describe_error(exc: BaseException) -> str:
    """Переводит исключение в понятное сообщение без технических деталей."""
    if isinstance(exc, DocumentLoadError):
        return str(exc)
    if isinstance(exc, ChatServiceError):
        return exc.user_message
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return "Не удалось получить доступ к сервису ИИ. Сообщите администратору бота."
    if isinstance(exc, openai.RateLimitError):
        return "Сервис ИИ сейчас перегружен или исчерпан лимит запросов. Попробуйте чуть позже."
    if isinstance(exc, openai.NotFoundError):
        return "Модель ИИ недоступна. Сообщите администратору бота."
    if isinstance(exc, openai.BadRequestError):
        return "Сервис ИИ не смог обработать запрос. Попробуйте переформулировать или сократить его."
    if isinstance(exc, openai.APIConnectionError):  # включает и таймауты
        return "Не удалось связаться с сервисом ИИ. Проверьте соединение и попробуйте ещё раз."
    if isinstance(exc, openai.OpenAIError):
        return "Сервис ИИ вернул ошибку. Попробуйте ещё раз чуть позже."
    if isinstance(exc, VectorStoreError):
        return "Ошибка хранилища документов. Попробуйте ещё раз чуть позже."
    if isinstance(exc, sqlite3.Error):
        return "Ошибка базы данных документов. Попробуйте ещё раз чуть позже."
    if isinstance(exc, TelegramNetworkError):
        return "Проблема с сетью при обращении к Telegram. Попробуйте ещё раз."
    if isinstance(exc, TelegramAPIError):
        return "Telegram не смог выполнить операцию. Попробуйте ещё раз."
    if isinstance(exc, (asyncio.TimeoutError, ConnectionError)):
        return "Сетевая ошибка. Попробуйте ещё раз."
    return "Что-то пошло не так. Попробуйте ещё раз чуть позже."


def _user_id(message: Message) -> int:
    assert message.from_user is not None  # в личном чате отправитель всегда известен
    return message.from_user.id


def _format_date(iso_value: str) -> str:
    try:
        return datetime.fromisoformat(iso_value).strftime("%d.%m.%Y %H:%M") + " UTC"
    except ValueError:
        return iso_value


def _pick_document(
    app: App, user_id: int, args: str | None
) -> tuple[DocumentRecord | None, str | None]:
    """Находит документ по номеру из аргумента команды. Возвращает (документ, текст ошибки)."""
    documents = app.registry.list_documents(user_id)
    if not documents:
        return None, "У вас пока нет загруженных документов. Отправьте файл PDF, DOCX или TXT."

    first = (args or "").split()[:1]
    if not first or not first[0].isdigit():
        return None, "Укажите номер документа, например: /use_document 1. Номера — в /documents."

    number = int(first[0])
    if not 1 <= number <= len(documents):
        return None, f"Документа с номером {number} нет. Посмотрите список: /documents."
    return documents[number - 1], None


# ---------- команды ----------


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(START_TEXT)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(HELP_TEXT)


@router.message(Command("clear_chat"))
async def cmd_clear_chat(message: Message, app: App) -> None:
    user_id = _user_id(message)
    # Под замком, чтобы не столкнуться с ответом или резюмированием, которые идут прямо сейчас
    async with app.locks.get(user_id):
        app.chat.clear_history(user_id)
    logger.info("История диалога очищена: user=%s", user_id)
    await message.answer("История текущего диалога очищена")


@router.message(Command("documents"))
async def cmd_documents(message: Message, app: App) -> None:
    user_id = _user_id(message)
    documents = app.registry.list_documents(user_id)
    if not documents:
        await message.answer("У вас пока нет загруженных документов. Отправьте файл PDF, DOCX или TXT.")
        return

    active_id = app.registry.get_active_doc_id(user_id)
    lines = ["Ваши документы:"]
    for number, doc in enumerate(documents, start=1):
        marker = " ← активный" if doc.id == active_id else ""
        lines.append(
            f"{number}. {doc.original_filename} — {_format_date(doc.uploaded_at)}, "
            f"чанков: {doc.chunks_count}, id: {doc.id[:8]}{marker}"
        )
    lines.append("")
    lines.append(
        "Поиск идёт только по активному документу. Вернуться ко всем — /all_documents."
        if active_id
        else "Поиск идёт по всем документам. Ограничить поиск одним — /use_document <номер>."
    )
    await send_long(message, "\n".join(lines))


@router.message(Command("use_document"))
async def cmd_use_document(message: Message, command: CommandObject, app: App) -> None:
    user_id = _user_id(message)
    doc, error = _pick_document(app, user_id, command.args)
    if doc is None:
        await message.answer(error or "Документ не найден.")
        return
    app.registry.set_active_doc_id(user_id, doc.id)
    await message.answer(
        f"Активный документ: {doc.original_filename}. Теперь отвечаю только по нему. "
        "Чтобы искать по всем документам, отправьте /all_documents."
    )


@router.message(Command("all_documents"))
async def cmd_all_documents(message: Message, app: App) -> None:
    app.registry.set_active_doc_id(_user_id(message), None)
    await message.answer("Выбор документа отменён: теперь ищу по всем вашим документам.")


@router.message(Command("delete_document"))
async def cmd_delete_document(message: Message, command: CommandObject, app: App) -> None:
    doc, error = _pick_document(app, _user_id(message), command.args)
    if doc is None:
        await message.answer(error or "Документ не найден.")
        return

    # Подтверждение через inline-кнопки; в callback_data лежит только doc_id (uuid) — до лимита 64 байта далеко
    keyboard = InlineKeyboardBuilder()
    keyboard.button(text="Удалить", callback_data=f"del:yes:{doc.id}")
    keyboard.button(text="Отмена", callback_data="del:no")
    keyboard.adjust(2)
    await message.answer(
        f"Удалить документ «{doc.original_filename}»? "
        "Будут удалены файл, индекс и запись о нём. Это нельзя отменить.",
        reply_markup=keyboard.as_markup(),
    )


@router.callback_query(F.data.startswith("del:"))
async def on_delete_confirmation(callback: CallbackQuery, app: App) -> None:
    user_id = callback.from_user.id
    parts = (callback.data or "").split(":")

    if len(parts) == 2 and parts[1] == "no":
        text = "Удаление отменено."
    elif len(parts) == 3 and parts[1] == "yes":
        try:
            # doc_id ищется среди документов ИМЕННО этого пользователя, чужой удалить нельзя
            deleted = await asyncio.to_thread(app.documents.delete, user_id, parts[2])
        except Exception as exc:
            logger.exception("Ошибка при удалении документа (user=%s)", user_id)
            text = describe_error(exc)
        else:
            text = "Документ удалён вместе с индексом." if deleted else "Документ не найден: возможно, он уже удалён."
    else:
        text = "Некорректный запрос."

    await callback.answer()
    if isinstance(callback.message, Message):
        try:
            await callback.message.edit_text(text)  # заодно убирает кнопки
        except TelegramAPIError:
            logger.warning("Не удалось обновить сообщение с подтверждением", exc_info=True)


@router.message(F.text.startswith("/"))
async def on_unknown_command(message: Message) -> None:
    await message.answer("Не знаю такой команды. Список команд — /help.")


# ---------- документы ----------


@router.message(F.document)
async def on_document(message: Message, app: App) -> None:
    user_id = _user_id(message)
    tg_doc = message.document
    assert tg_doc is not None
    settings = app.settings

    safe_name = sanitize_filename(tg_doc.file_name or "document")
    if Path(safe_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        await message.answer("Этот формат не поддерживается. Отправьте файл PDF, DOCX или TXT.")
        return
    if tg_doc.file_size == 0:
        await message.answer("Файл пустой. Отправьте документ с текстом.")
        return
    if tg_doc.file_size is not None and tg_doc.file_size > settings.max_document_bytes:
        await message.answer(f"Файл слишком большой. Максимальный размер — {settings.max_document_size_mb} МБ.")
        return

    status = await message.answer("Файл получен. Извлекаю текст и создаю индекс…")

    async def update_status(text: str) -> None:
        try:
            await status.edit_text(text)
        except TelegramAPIError:
            logger.warning("Не удалось обновить статус обработки", exc_info=True)

    path: Path | None = None
    try:
        doc_id, safe_name, path = app.documents.new_upload(user_id, tg_doc.file_name or "document")
        async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
            await message.bot.download(tg_doc, destination=path)
            record = await app.documents.index_file(user_id, doc_id, safe_name, path, progress=update_status)
    except Exception as exc:
        logger.exception("Не удалось обработать документ (user=%s)", user_id)
        if path is not None:
            path.unlink(missing_ok=True)  # на случай, если упало ещё при скачивании
        await update_status(describe_error(exc))
        return

    try:
        await status.delete()
    except TelegramAPIError:
        pass
    await message.answer(
        f"Готово: {record.original_filename} добавлен в долгосрочную память. "
        f"Создано чанков: {record.chunks_count}. Теперь можно задавать вопросы."
    )


# ---------- обычные сообщения ----------


@router.message(F.text)
async def on_text(message: Message, app: App) -> None:
    user_id = _user_id(message)
    question = (message.text or "").strip()
    if not question:
        return

    # Сообщения одного пользователя обрабатываем по очереди, чтобы не перемешать его историю
    async with app.locks.get(user_id):
        try:
            async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
                answer = await app.chat.answer(user_id, question)
        except Exception as exc:
            logger.exception("Не удалось ответить на сообщение (user=%s)", user_id)
            await message.answer(describe_error(exc))
            return

        await send_long(message, answer)
        # Резюмирование после отправки ответа: пользователь не ждёт лишний запрос к модели
        await app.chat.maybe_summarize(user_id)


@router.message()
async def on_other_content(message: Message) -> None:
    await message.answer("Я работаю с текстом и документами PDF, DOCX или TXT. Отправьте одно из них.")


@router.errors()
async def on_error(event: ErrorEvent) -> bool:
    """Последняя линия обороны: пишем traceback в лог, пользователю — короткое сообщение."""
    logger.error("Необработанная ошибка при обработке обновления", exc_info=event.exception)
    update = event.update
    target = update.message or (update.callback_query.message if update.callback_query else None)
    if isinstance(target, Message):
        try:
            await target.answer(describe_error(event.exception))
        except TelegramAPIError:
            logger.warning("Не удалось отправить сообщение об ошибке", exc_info=True)
    return True


# ---------- запуск ----------


async def main() -> None:
    setup_logging()
    try:
        settings = load_settings()
    except ConfigError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    setup_logging(settings.log_level, secrets=[settings.bot_token, settings.openai_api_key])
    logger.info("Запуск бота (модель чата: %s, эмбеддинги: %s)", settings.chat_model, settings.embed_model)

    try:
        store = VectorStore(settings.chroma_path)
        registry = DocumentRegistry(settings.db_path)
    except (VectorStoreError, sqlite3.Error):
        logger.error("Не удалось открыть хранилища данных (подробности выше).")
        sys.exit(1)

    client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=60.0, max_retries=2)
    embeddings = EmbeddingService(client, settings.embed_model)
    documents = DocumentService(settings, registry, store, embeddings)
    documents.cleanup_unfinished()

    memory = ShortMemory(
        settings.short_memory_messages,
        settings.short_memory_summary_after,
        SUMMARY_KEEP_MESSAGES,
    )
    rag = RagService(embeddings, store, settings.rag_max_distance)
    chat = ChatService(settings, client, memory, rag, registry)
    app = App(settings=settings, registry=registry, documents=documents, chat=chat)

    try:
        bot = Bot(token=settings.bot_token)
    except TokenValidationError:
        logger.error("BOT_TOKEN имеет неверный формат. Проверьте значение в .env.")
        sys.exit(1)

    dispatcher = Dispatcher(app=app)
    dispatcher.include_router(router)

    try:
        me = await bot.get_me()
        logger.info("Бот запущен: @%s", me.username)
        await bot.set_my_commands(
            [
                BotCommand(command="start", description="Приветствие"),
                BotCommand(command="help", description="Справка"),
                BotCommand(command="documents", description="Мои документы"),
                BotCommand(command="use_document", description="Искать только в документе"),
                BotCommand(command="all_documents", description="Искать по всем документам"),
                BotCommand(command="delete_document", description="Удалить документ"),
                BotCommand(command="clear_chat", description="Очистить историю диалога"),
            ]
        )
        await dispatcher.start_polling(bot, allowed_updates=dispatcher.resolve_used_update_types())
    except TelegramUnauthorizedError:
        logger.error("Telegram отклонил BOT_TOKEN: токен неверный или отозван.")
        sys.exit(1)
    except TelegramNetworkError:
        logger.error("Нет связи с Telegram. Проверьте интернет-соединение.", exc_info=True)
        sys.exit(1)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Бот остановлен.")
