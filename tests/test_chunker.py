"""Тесты чанкинга: границы, перекрытие, защита от зацикливания."""
import pytest

from services.chunker import chunk_text, normalize_text


def test_empty_and_whitespace_text_gives_no_chunks():
    assert chunk_text("", 500, 75) == []
    assert chunk_text("   \n\n \t ", 500, 75) == []


def test_short_text_is_single_chunk():
    chunks = chunk_text("Привет, мир!", 500, 75)
    assert len(chunks) == 1
    assert chunks[0].text == "Привет, мир!"
    assert (chunks[0].chunk_index, chunks[0].start_char, chunks[0].end_char) == (0, 0, 12)


def test_chunks_respect_size_and_have_sequential_indexes():
    text = " ".join(f"слово{i}" for i in range(400))
    chunks = chunk_text(text, 200, 30)
    assert len(chunks) > 1
    assert all(0 < len(c.text) <= 200 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_offsets_point_to_normalized_text():
    text = "Первое предложение. Второе предложение.\n\nНовый абзац с текстом. " * 20
    normalized = normalize_text(text)
    for chunk in chunk_text(text, 150, 20):
        assert normalized[chunk.start_char : chunk.end_char] == chunk.text


def test_consecutive_chunks_overlap_and_cover_whole_text():
    text = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_text(text, 100, 25)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.start_char < prev.end_char  # есть перекрытие
        assert nxt.start_char > prev.start_char  # но есть и прогресс
    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(normalize_text(text))


def test_prefers_paragraph_boundary():
    first = "А" * 60
    second = "Б" * 60
    chunks = chunk_text(f"{first}\n\n{second}", 100, 10)
    assert chunks[0].text == first


def test_prefers_sentence_boundary():
    text = "Это первое предложение довольно длинное. Это второе предложение, оно не поместится целиком."
    chunks = chunk_text(text, 60, 5)
    assert chunks[0].text == "Это первое предложение довольно длинное."


def test_hard_split_for_text_without_spaces():
    chunks = chunk_text("x" * 1000, 100, 10)
    assert all(len(c.text) <= 100 for c in chunks)
    assert chunks[-1].end_char == 1000


@pytest.mark.parametrize("overlap", [-5, 0, 100, 150, 10_000])
def test_invalid_overlap_does_not_hang(overlap):
    text = "слово " * 500
    chunks = chunk_text(text, 100, overlap)
    assert chunks
    assert all(c.text.strip() for c in chunks)


def test_tiny_chunk_size_terminates():
    chunks = chunk_text("abc def ghi", 1, 5)
    assert chunks
    assert all(len(c.text) == 1 for c in chunks)


def test_invalid_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("text", 0, 0)


def test_normalize_collapses_whitespace_but_keeps_paragraphs():
    text = "Строка   с    пробелами\t\tи табами.\r\n\r\n\r\n\r\nВторой   абзац."
    assert normalize_text(text) == "Строка с пробелами и табами.\n\nВторой абзац."


def test_no_empty_chunks_for_text_with_many_blank_lines():
    text = "Абзац один.\n\n\n\n\n\n\n\nАбзац два.\n\n\n\n\n\nАбзац три."
    chunks = chunk_text(text, 20, 5)
    assert all(c.text.strip() for c in chunks)
