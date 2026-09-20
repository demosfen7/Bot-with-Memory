"""Тесты кратковременной памяти: изоляция пользователей, FIFO, резюмирование."""
from services.short_memory import ShortMemory


def test_history_is_isolated_per_user():
    memory = ShortMemory(window=10, summary_after=20)
    memory.add_user_message(1, "привет от первого")
    memory.add_user_message(2, "привет от второго")
    assert [m.content for m in memory.get_history(1)] == ["привет от первого"]
    assert [m.content for m in memory.get_history(2)] == ["привет от второго"]


def test_window_limits_history_sent_to_model():
    memory = ShortMemory(window=4, summary_after=0)
    for i in range(10):
        memory.add_user_message(1, f"m{i}")
    assert [m.content for m in memory.get_history(1)] == ["m6", "m7", "m8", "m9"]


def test_fifo_when_summary_disabled():
    memory = ShortMemory(window=3, summary_after=0)
    for i in range(5):
        memory.add_user_message(1, f"m{i}")
    assert [m.content for m in memory.get_history(1)] == ["m2", "m3", "m4"]
    assert not memory.needs_summary(1)


def test_summary_flow_keeps_last_messages():
    memory = ShortMemory(window=10, summary_after=20, keep_after_summary=6)
    for i in range(21):
        memory.add_user_message(1, f"m{i}")
    assert memory.needs_summary(1)

    batch = memory.messages_to_summarize(1)
    assert len(batch) == 15
    memory.apply_summary(1, "краткое резюме", len(batch))

    assert memory.get_summary(1) == "краткое резюме"
    assert [m.content for m in memory.get_history(1)] == [f"m{i}" for i in range(15, 21)]
    assert not memory.needs_summary(1)


def test_discard_last_user_message_only_removes_unanswered_question():
    memory = ShortMemory(window=10, summary_after=0)
    memory.add_user_message(1, "вопрос")
    memory.discard_last_user_message(1)
    assert memory.get_history(1) == []

    memory.add_user_message(1, "вопрос")
    memory.add_assistant_message(1, "ответ")
    memory.discard_last_user_message(1)  # последняя реплика — ответ, ничего не удаляется
    assert len(memory.get_history(1)) == 2


def test_clear_affects_only_one_user():
    memory = ShortMemory(window=10, summary_after=20)
    memory.add_user_message(1, "a")
    memory.add_user_message(2, "b")
    memory.apply_summary(1, "резюме 1", 0)
    memory.apply_summary(2, "резюме 2", 0)

    memory.clear(1)

    assert memory.get_history(1) == []
    assert memory.get_summary(1) == ""
    assert len(memory.get_history(2)) == 1
    assert memory.get_summary(2) == "резюме 2"
