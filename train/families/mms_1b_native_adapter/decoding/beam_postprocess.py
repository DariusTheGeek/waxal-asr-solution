#!/usr/bin/env python3
"""WAXAL2-compatible beam labels and fold-safe Lingala text reconstruction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
import unicodedata
from typing import Iterable


WAXAL2_BEAM_SHA256 = "e49bd40fdd77ce9a69c760d9dbac5299854a0020c277e60381e49532df0597cc"
WAXAL2_PUNCT_SHA256 = "9c63830870151792548dd03d0c72e1543193d3820a09ea4532233510bace8397"
WAXAL2_TRUECASE_SHA256 = "6971c9e7dd1d9eb5bd7286d4141f4112caaf930793ae8b084d384ddf19c70e95"

WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)
SENT_END = ".!?"
_ALPHA = "a-zà-ÿāăąćčđēėęěğīįķĺļłńņňŋōőœŕřśşšţūůűųźżžþ"
_CAPFIRST = re.compile(r"(^|[%s]\s+)([%s])" % (re.escape(SENT_END), _ALPHA))
_CORE = re.compile(r"[^\w']", re.UNICODE)


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFC", "" if value is None else str(value))
    return " ".join(text.split())


def labels_from_vocab(vocabulary: dict[str, int]) -> list[str]:
    """Return the exact pyctcdecode index-to-label mapping used by WAXAL2."""

    if vocabulary.get("<pad>") != 0 or vocabulary.get("<unk>") != 1:
        raise ValueError("pad/unknown token IDs must be 0/1")
    if vocabulary.get("|") != 2:
        raise ValueError("word delimiter ID must be 2")
    if sorted(int(value) for value in vocabulary.values()) != list(
        range(len(vocabulary))
    ):
        raise ValueError("vocabulary IDs must be unique and contiguous")
    inverse = {int(value): str(key) for key, value in vocabulary.items()}
    labels: list[str] = []
    for index in range(len(inverse)):
        token = inverse[index]
        if token == "<pad>":
            labels.append("")
        elif token == "|":
            labels.append(" ")
        elif token == "<unk>":
            labels.append("⁇")
        else:
            labels.append(token)
    return labels


def greedy_ctc(values: Iterable[int], vocabulary: dict[str, int]) -> str:
    inverse = {int(value): str(key) for key, value in vocabulary.items()}
    tokens: list[str] = []
    previous: int | None = None
    for raw_value in values:
        value = int(raw_value)
        if value == 0:
            previous = value
            continue
        if value == previous:
            continue
        previous = value
        token = inverse.get(value, "<unk>")
        if token not in {"<pad>", "<unk>"}:
            tokens.append(token)
    return normalize_text("".join(tokens).replace("|", " "))


def sentence_case(value: object) -> str:
    characters = list(normalize_text(value))
    for index, character in enumerate(characters):
        if character.isalpha():
            characters[index] = character.upper()
            break
    return "".join(characters)


def _tokens_with_marks(text: str) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    for match in WORD.finditer(text):
        word = match.group(0).lower()
        tail = text[match.end() : match.end() + 3]
        mark = ""
        for character in tail:
            if character in ",.":
                mark = character
                break
            if not character.isspace() and character not in "\"'’)":
                break
        output.append((word, mark))
    return output


class PunctModel:
    """Exact WAXAL2 comma/period count model, fitted on training text only."""

    def __init__(self) -> None:
        self.bi: dict[tuple[str, str], Counter[str]] = {}
        self.uni: dict[str, Counter[str]] = {}
        self.final: Counter[str] = Counter()
        self.final_word: dict[str, Counter[str]] = {}

    def fit(self, texts: Iterable[str]) -> "PunctModel":
        for text in texts:
            tokens = _tokens_with_marks(text)
            if not tokens:
                continue
            for index, (word, mark) in enumerate(tokens):
                if index == len(tokens) - 1:
                    self.final[mark] += 1
                    self.final_word.setdefault(word, Counter())[mark] += 1
                    continue
                following = tokens[index + 1][0]
                self.bi.setdefault((word, following), Counter())[mark] += 1
                self.uni.setdefault(word, Counter())[mark] += 1
        return self

    @staticmethod
    def _pick(counter: Counter[str], minimum: int, ratio: float) -> str:
        total = sum(counter.values())
        if total < minimum:
            return ""
        mark, count = counter.most_common(1)[0]
        if mark == "" or count / total < ratio:
            return ""
        return mark

    def terminal_only(
        self, text: str, *, final_minimum: int = 5, final_ratio: float = 0.5
    ) -> str:
        """Reproduce WAXAL2's deployed terminal-only punctuation policy."""

        words = normalize_text(text).split()
        if not words:
            return normalize_text(text)
        counts = self.final_word.get(words[-1].lower())
        if counts and sum(counts.values()) >= final_minimum:
            mark = self._pick(counts, final_minimum, final_ratio)
        else:
            total = sum(self.final.values())
            mark = "." if total and self.final["."] / total >= final_ratio else ""
        return normalize_text(text) + mark

    def apply(
        self,
        text: str,
        *,
        bi_minimum: int = 3,
        bi_ratio: float = 0.55,
        uni_minimum: int = 20,
        uni_ratio: float = 0.55,
        final_minimum: int = 5,
        final_ratio: float = 0.5,
    ) -> str:
        words = normalize_text(text).split()
        if not words:
            return normalize_text(text)
        output: list[str] = []
        for index, word in enumerate(words):
            output.append(word)
            if index == len(words) - 1:
                break
            key = (word.lower(), words[index + 1].lower())
            mark = ""
            if key in self.bi:
                mark = self._pick(self.bi[key], bi_minimum, bi_ratio)
            if not mark and word.lower() in self.uni:
                mark = self._pick(self.uni[word.lower()], uni_minimum, uni_ratio)
            if mark:
                output[-1] += mark
        base = " ".join(output)
        return self.terminal_only(
            base, final_minimum=final_minimum, final_ratio=final_ratio
        )


def learn_proper(
    train_texts: Iterable[str], *, minimum_occurrences: int = 5, cap_fraction: float = 0.5
) -> set[str]:
    total: Counter[str] = Counter()
    capitalized: Counter[str] = Counter()
    for text in train_texts:
        words = text.split()
        for index, word in enumerate(words):
            core = _CORE.sub("", word)
            if not core or not core[0].isalpha():
                continue
            previous = words[index - 1] if index > 0 else "."
            if index == 0 or previous[-1:] in SENT_END:
                continue
            total[core.lower()] += 1
            if core[0].isupper():
                capitalized[core.lower()] += 1
    return {
        word
        for word in total
        if total[word] >= minimum_occurrences
        and capitalized[word] / total[word] >= cap_fraction
    }


def truecase(text: str, proper: set[str]) -> str:
    text = _CAPFIRST.sub(lambda match: match.group(1) + match.group(2).upper(), text)
    output: list[str] = []
    for word in text.split():
        core = _CORE.sub("", word).lower()
        if core in proper and word[:1].islower():
            index = next(
                (offset for offset, character in enumerate(word) if character.isalpha()),
                None,
            )
            if index is not None:
                word = word[:index] + word[index].upper() + word[index + 1 :]
        output.append(word)
    return " ".join(output)


def always_period(text: str) -> str:
    text = normalize_text(text)
    if not text or text[-1:] in SENT_END:
        return text
    return text + "."


@dataclass(frozen=True)
class PostprocessAssets:
    punctuation: PunctModel
    proper: set[str]
    training_rows: int

    @classmethod
    def fit(cls, train_texts: list[str]) -> "PostprocessAssets":
        texts = [normalize_text(text) for text in train_texts if normalize_text(text)]
        if not texts:
            raise ValueError("training text is empty")
        return cls(
            punctuation=PunctModel().fit(texts),
            proper=learn_proper(texts),
            training_rows=len(texts),
        )

    def provenance(self) -> dict[str, object]:
        total = sum(self.punctuation.final.values())
        return {
            "training_rows": self.training_rows,
            "proper_lexicon_size": len(self.proper),
            "proper_lexicon": sorted(self.proper),
            "terminal_counts": dict(sorted(self.punctuation.final.items())),
            "terminal_period_fraction": (
                self.punctuation.final["."] / total if total else 0.0
            ),
            "final_word_types": len(self.punctuation.final_word),
            "constants": {
                "minimum_occurrences": 5,
                "cap_fraction": 0.5,
                "sentence_end": SENT_END,
            },
        }


POLICIES = (
    "raw",
    "sentence_case",
    "truecase",
    "waxal2_terminal_sentence_case",
    "waxal2_terminal_truecase",
    "always_period_sentence_case",
    "always_period_truecase",
    "full_punct_sentence_case",
    "full_punct_truecase",
)


def apply_policy(text: str, policy: str, assets: PostprocessAssets) -> str:
    text = normalize_text(text)
    if policy == "raw":
        return text
    if policy == "sentence_case":
        return sentence_case(text)
    if policy == "truecase":
        return truecase(text, assets.proper)
    if policy == "waxal2_terminal_sentence_case":
        return sentence_case(assets.punctuation.terminal_only(text))
    if policy == "waxal2_terminal_truecase":
        return truecase(assets.punctuation.terminal_only(text), assets.proper)
    if policy == "always_period_sentence_case":
        return sentence_case(always_period(text))
    if policy == "always_period_truecase":
        return truecase(always_period(text), assets.proper)
    if policy == "full_punct_sentence_case":
        return sentence_case(assets.punctuation.apply(text))
    if policy == "full_punct_truecase":
        return truecase(assets.punctuation.apply(text), assets.proper)
    raise ValueError(f"unknown post-processing policy: {policy}")
