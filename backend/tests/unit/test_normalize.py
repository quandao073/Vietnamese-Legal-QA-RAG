from legalrag.ingest.normalize import normalize_text, strip_boilerplate, tokenize_for_bm25


def test_nfc_and_tone_normalization():
    # decomposed unicode -> NFC
    assert normalize_text("Việt") == "Việt"
    # old-style tone placement -> new style
    assert normalize_text("hoà bình, thuỵ sĩ, hoạ sĩ, Hoà") == "hòa bình, thụy sĩ, họa sĩ, Hòa"
    # tone stays on the second vowel when a consonant follows
    assert normalize_text("Khoản toàn thuyền") == "Khoản toàn thuyền"


def test_dashes_control_chars_and_spaces():
    assert normalize_text("a  –  b\u200b\u00a0c\t—\td") == "a - b c - d"


def test_markers_preserved():
    s = "Chương I\nĐiều 1. Phạm vi\n1. Khoản một\na) điểm a\nđ) điểm đ"
    assert normalize_text(s) == s


def test_blank_lines_collapsed():
    assert normalize_text("a\n\n\n\nb\n") == "a\n\nb"


def test_strip_boilerplate_removes_header_and_noi_nhan():
    text = (
        "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
        "Độc lập - Tự do - Hạnh phúc\n"
        "Điều 1. Nội dung\n"
        "Nội dung điều 1.\n"
        "3\n"
        "Nơi nhận:\n"
        "- Lưu VT.\n"
    )
    out = strip_boilerplate(text)
    assert out == "Điều 1. Nội dung\nNội dung điều 1."


def test_strip_boilerplate_keeps_text_without_pattern():
    text = "Điều 1. A\n1. Nội dung có 3 phần.\n2. Hai"
    assert strip_boilerplate(text) == text


def test_tokenize_for_bm25_lowercases_and_joins_compounds():
    out = tokenize_for_bm25("Thuế giá trị gia tăng là gì?")
    assert out == out.lower()
    assert "_" in out
    assert "  " not in out
