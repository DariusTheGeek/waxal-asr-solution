"""Text normalisation shared by every stage.

Kept in one place because fusion, scoring and submission writing must agree
byte for byte on what a token is; a divergence here shows up as an unexplained
score gap rather than an error.
"""
from __future__ import annotations

import re
import unicodedata

# Strip leading/trailing punctuation when comparing tokens, but never when
# emitting them -- the surface form is what gets scored.
TOKEN_EDGE = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)


def normalize_text(value: object) -> str:
    """NFC-normalise and collapse whitespace. Idempotent."""
    return " ".join(unicodedata.normalize("NFC", str(value or "")).split())


def token_key(token: str) -> str:
    """Comparison key for a token: case-folded, punctuation-stripped."""
    return TOKEN_EDGE.sub("", token.casefold())


def word_tokens(value: object) -> list[str]:
    return normalize_text(value).split()
