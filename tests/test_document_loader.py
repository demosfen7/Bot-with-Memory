"""Тесты безопасных имён файлов и чтения документов."""
import pytest

from services.document_loader import DocumentLoadError, extract_text, sanitize_filename


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd.txt", "passwd.txt"),
        ("..\\..\\Windows\\system32\\evil.docx", "evil.docx"),
        ("отчёт по проекту.PDF", "отчёт по проекту.pdf"),
        ("a<b>:c|d?.txt", "a_b_c_d_.txt"),
        ("", "document"),
        ("...", "document"),
    ],
)
def test_sanitize_filename(raw, expected):
    assert sanitize_filename(raw) == expected


def test_sanitize_filename_never_contains_path_separators():
    for raw in ["/etc/passwd", "a/b/c.txt", "..", "C:\\x\\y.txt", "x\x00y.txt"]:
        name = sanitize_filename(raw)
        assert "/" not in name and "\\" not in name and "\x00" not in name


def test_sanitize_filename_limits_length_and_keeps_extension():
    name = sanitize_filename("а" * 500 + ".pdf")
    assert len(name) <= 100
    assert name.endswith(".pdf")


def test_extract_txt_utf8_and_cp1251(tmp_path):
    utf8 = tmp_path / "a.txt"
    utf8.write_text("Привет, мир", encoding="utf-8")
    assert extract_text(utf8) == "Привет, мир"

    legacy = tmp_path / "b.txt"
    legacy.write_bytes("Привет, мир".encode("cp1251"))
    assert extract_text(legacy) == "Привет, мир"


def test_empty_txt_raises(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("  \n\n ", encoding="utf-8")
    with pytest.raises(DocumentLoadError):
        extract_text(empty)


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "x.exe"
    path.write_bytes(b"MZ")
    with pytest.raises(DocumentLoadError):
        extract_text(path)


def test_broken_pdf_and_docx_raise_friendly_error(tmp_path):
    for name in ("broken.pdf", "broken.docx"):
        path = tmp_path / name
        path.write_bytes(b"this is not a real document")
        with pytest.raises(DocumentLoadError):
            extract_text(path)
