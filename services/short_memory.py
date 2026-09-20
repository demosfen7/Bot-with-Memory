"""Кратковременная память: история диалога и резюме по каждому пользователю (в оперативной памяти)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    role: str  # "user" или "assistant"
    content: str


class ShortMemory:
    """Хранит реплики в deque на каждого user_id.

    Три числа, которые нужно различать:
    - window — сколько последних реплик уходит в модель вместе с новым вопросом;
    - summary_after — когда реплик накопилось больше, старые сворачиваются в резюме
      (0 — резюмирование выключено);
    - keep_after_summary — сколько последних реплик остаётся после резюмирования.

    Ёмкость deque = max(window, summary_after + 2), иначе счётчик никогда не дошёл бы
    до порога резюмирования. Когда deque заполнен, самая старая реплика вытесняется (FIFO).
    """

    def __init__(self, window: int, summary_after: int, keep_after_summary: int = 6) -> None:
        self._window = window
        self._summary_after = summary_after
        self._keep = keep_after_summary
        self._capacity = max(window, summary_after + 2) if summary_after > 0 else window
        self._messages: dict[int, deque[Message]] = {}
        self._summaries: dict[int, str] = {}

    # ---- запись ----

    def _queue(self, user_id: int) -> deque[Message]:
        if user_id not in self._messages:
            self._messages[user_id] = deque(maxlen=self._capacity)
        return self._messages[user_id]

    def add_user_message(self, user_id: int, content: str) -> None:
        self._queue(user_id).append(Message("user", content))

    def add_assistant_message(self, user_id: int, content: str) -> None:
        self._queue(user_id).append(Message("assistant", content))

    def discard_last_user_message(self, user_id: int) -> None:
        """Откатывает реплику пользователя, если ответить на неё не удалось."""
        queue = self._messages.get(user_id)
        if queue and queue[-1].role == "user":
            queue.pop()

    # ---- чтение ----

    def get_history(self, user_id: int) -> list[Message]:
        """Последние `window` реплик пользователя (включая самую свежую)."""
        queue = self._messages.get(user_id)
        if not queue:
            return []
        return list(queue)[-self._window :]

    def get_summary(self, user_id: int) -> str:
        return self._summaries.get(user_id, "")

    # ---- резюмирование ----

    def needs_summary(self, user_id: int) -> bool:
        queue = self._messages.get(user_id)
        return self._summary_after > 0 and queue is not None and len(queue) > self._summary_after

    def messages_to_summarize(self, user_id: int) -> list[Message]:
        """Все реплики, кроме последних `keep_after_summary`."""
        queue = self._messages.get(user_id)
        if not queue:
            return []
        return list(queue)[: max(0, len(queue) - self._keep)]

    def apply_summary(self, user_id: int, summary: str, summarized_count: int) -> None:
        """Сохраняет резюме и убирает из истории реплики, которые в него вошли."""
        queue = self._queue(user_id)
        for _ in range(min(summarized_count, len(queue))):
            queue.popleft()
        self._summaries[user_id] = summary

    # ---- очистка ----

    def clear(self, user_id: int) -> None:
        """Удаляет историю и резюме только этого пользователя."""
        self._messages.pop(user_id, None)
        self._summaries.pop(user_id, None)
