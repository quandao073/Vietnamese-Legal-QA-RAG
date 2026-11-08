"""Text normalization for Vietnamese legal text (spec §2.4)."""

from __future__ import annotations

import re
import unicodedata

# Tone placement for the open syllables "oa/oe/uy": the corpus convention we standardise to
# puts the tone on the first vowel ("hòa", "thụy", "họa"). The alternative style ("hoà") is
# only rewritten when the vowel pair ends the word; with a following consonant ("Khoản",
# "toàn") both styles agree and nothing changes.
_TONE_PAIRS = {
    "oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa",
    "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe",
    "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy",
}  # fmt: skip
_OLD_TO_NEW = dict(_TONE_PAIRS)
_OLD_TO_NEW.update({k[0].upper() + k[1]: v[0].upper() + v[1] for k, v in _TONE_PAIRS.items()})
_TONE_RE = re.compile("(" + "|".join(map(re.escape, _OLD_TO_NEW)) + r")(?![^\W\d_])")

_CONTROL_CODEPOINTS = [*range(0, 9), 11, 12, *range(14, 32), 0x7F, 0x200B, 0x200C, 0x200D, 0xFEFF]
_CONTROL_RE = re.compile("[" + "".join(map(chr, _CONTROL_CODEPOINTS)) + "]")
_DASH_RE = re.compile(r"[‒–—―−]")
_SPACE_CODEPOINTS = [0x20, 0x09, 0xA0, *range(0x2000, 0x200B), 0x202F, 0x205F, 0x3000]
_SPACES_RE = re.compile("[" + "".join(map(chr, _SPACE_CODEPOINTS)) + "]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")

_BOILERPLATE_LINES = (
    re.compile(r"^\s*CỘNG\s+HÒA\s+XÃ\s+HỘI\s+CHỦ\s+NGHĨA\s+VIỆT\s+NAM\s*$", re.IGNORECASE),
    re.compile(r"^\s*Độc\s+lập\s*-\s*Tự\s+do\s*-\s*Hạnh\s+phúc\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d{1,4}\s*$"),  # bare page numbers
)
_NOI_NHAN_RE = re.compile(r"^\s*Nơi nhận\s*:", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """NFC, unified tone marks, no control chars, ASCII dashes, collapsed whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TONE_RE.sub(lambda m: _OLD_TO_NEW[m.group(1)], text)
    text = _DASH_RE.sub("-", text)
    lines = [_SPACES_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def strip_boilerplate(text: str) -> str:
    """Drop national header lines, page numbers and the trailing 'Nơi nhận:' block."""
    out: list[str] = []
    for line in text.split("\n"):
        if _NOI_NHAN_RE.match(line):
            break
        if any(p.match(line) for p in _BOILERPLATE_LINES):
            continue
        out.append(line)
    return "\n".join(out).strip()


def tokenize_for_bm25(text: str) -> str:
    """Word-segment with pyvi (compounds joined by '_'), lowercase, single-spaced."""
    from pyvi import ViTokenizer  # heavy import kept local

    return " ".join(ViTokenizer.tokenize(text.lower()).split())
