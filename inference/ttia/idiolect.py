"""Idiolect models for Test-Time Idiolect Adaptation (TTIA).

An idiolect is the individual language variety behind one voice: its
vocabulary, compounding habits and phrasing. Both classes here are built once
from the training transcripts, keyed by idiolect profile, and are consulted
at fusion time for the profile that stage 2 matched to each test clip.

``BoundaryModel``
    Splits compounds and merges adjacent words when the matched profile's
    training text clearly prefers the other convention. Word-boundary habits
    are the single largest source of inter-annotator disagreement in the
    Lingala labels.

``TextLanguageModel``
    A small count LM over the matched profile's training text, backed off to
    the global text. Used only to rank already-decoded candidate transcripts;
    it never generates text.

Both consume training text only. A profile key with no training text falls
back to the global counts, so clips whose profile never appears in the
labelled data are handled by the same code path with no special casing.
"""
from __future__ import annotations

import math
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inference.text import normalize_text  # noqa: E402


def raw_text(text: object, *, casefold: bool = False) -> str:
    value = normalize_text(text)
    return value.casefold() if casefold else value


def edge_parts(token: str) -> tuple[str, str, str]:
    start = 0
    end = len(token)
    while start < end and unicodedata.category(token[start])[0] in {"P", "S"}:
        start += 1
    while end > start and unicodedata.category(token[end - 1])[0] in {"P", "S"}:
        end -= 1
    return token[:start], token[start:end], token[end:]


def lexical_tokens(text: object) -> list[str]:
    output: list[str] = []
    for token in raw_text(text, casefold=True).split():
        core = edge_parts(token)[1]
        if core:
            output.append(core)
    return output


def raw_tokens(text: object) -> list[str]:
    return raw_text(text, casefold=True).split()


def match_initial_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    first_alpha = next((char for char in source if char.isalpha()), "")
    if first_alpha and first_alpha.isupper():
        chars = list(target)
        for index, char in enumerate(chars):
            if char.isalpha():
                chars[index] = char.upper()
                break
        return "".join(chars)
    return target.casefold()


@dataclass(frozen=True)
class BoundaryConfig:
    scope: str
    minimum_support: int
    preference_ratio: float


class BoundaryModel:
    def __init__(self, profiles: Iterable[str], texts: Iterable[str]) -> None:
        self.global_unigram: Counter[str] = Counter()
        self.global_bigram: Counter[tuple[str, str]] = Counter()
        self.profile_unigram: dict[str, Counter[str]] = defaultdict(Counter)
        self.profile_bigram: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
        for profile, text in zip(profiles, texts, strict=True):
            tokens = lexical_tokens(text)
            self.global_unigram.update(tokens)
            self.global_bigram.update(zip(tokens, tokens[1:]))
            self.profile_unigram[profile].update(tokens)
            self.profile_bigram[profile].update(zip(tokens, tokens[1:]))

    def _counts(
        self, profile: str, scope: str
    ) -> tuple[Counter[str], Counter[tuple[str, str]]]:
        if scope == "profile" and profile in self.profile_unigram:
            return self.profile_unigram[profile], self.profile_bigram[profile]
        return self.global_unigram, self.global_bigram

    def normalize(self, text: str, profile: str, config: BoundaryConfig) -> str:
        unigram, bigram = self._counts(profile, config.scope)
        expanded: list[str] = []

        # First split compounds whose separated form clearly dominates in the
        # matched profile's training labels.
        for token in raw_text(text).split():
            prefix, core, suffix = edge_parts(token)
            lower = core.casefold()
            best: tuple[float, int, int, str, str] | None = None
            if lower.isalpha() and len(lower) >= 4:
                for split_at in range(2, len(lower) - 1):
                    left = lower[:split_at]
                    right = lower[split_at:]
                    split_support = bigram[(left, right)]
                    joined_support = unigram[lower]
                    if (
                        split_support >= config.minimum_support
                        and split_support
                        >= config.preference_ratio * max(1, joined_support)
                    ):
                        candidate = (
                            split_support / max(1, joined_support),
                            split_support,
                            split_at,
                            left,
                            right,
                        )
                        if best is None or candidate > best:
                            best = candidate
            if best is None:
                expanded.append(token)
                continue
            _, _, _, left, right = best
            expanded.extend([prefix + match_initial_case(core, left), right + suffix])

        # Then merge adjacent words when the joined convention dominates.
        output: list[str] = []
        index = 0
        while index < len(expanded):
            if index + 1 < len(expanded):
                prefix_a, core_a, suffix_a = edge_parts(expanded[index])
                prefix_b, core_b, suffix_b = edge_parts(expanded[index + 1])
                left = core_a.casefold()
                right = core_b.casefold()
                joined = left + right
                joined_support = unigram[joined]
                split_support = bigram[(left, right)]
                if (
                    core_a
                    and core_b
                    and not suffix_a
                    and not prefix_b
                    and left.isalpha()
                    and right.isalpha()
                    and min(len(left), len(right)) >= 2
                    and joined_support >= config.minimum_support
                    and joined_support
                    >= config.preference_ratio * max(1, split_support)
                ):
                    output.append(
                        prefix_a + match_initial_case(core_a, joined) + suffix_b
                    )
                    index += 2
                    continue
            output.append(expanded[index])
            index += 1
        return " ".join(output)


class TextLanguageModel:
    """Small count LM used only to rank already-decoded text candidates."""

    def __init__(self, profiles: Iterable[str], texts: Iterable[str]) -> None:
        self.data: dict[bool, dict[str, object]] = {}
        rows = list(zip(profiles, texts, strict=True))
        for lexical in (False, True):
            global_unigram: Counter[str] = Counter()
            global_bigram: Counter[tuple[str, str]] = Counter()
            global_context: Counter[str] = Counter()
            profile_unigram: dict[str, Counter[str]] = defaultdict(Counter)
            profile_bigram: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
            profile_context: dict[str, Counter[str]] = defaultdict(Counter)
            for profile, text in rows:
                tokens = lexical_tokens(text) if lexical else raw_tokens(text)
                sequence = ["<s>", *tokens, "</s>"]
                global_unigram.update(sequence[1:])
                global_bigram.update(zip(sequence, sequence[1:]))
                global_context.update(sequence[:-1])
                profile_unigram[profile].update(sequence[1:])
                profile_bigram[profile].update(zip(sequence, sequence[1:]))
                profile_context[profile].update(sequence[:-1])
            self.data[lexical] = {
                "global_unigram": global_unigram,
                "global_bigram": global_bigram,
                "global_context": global_context,
                "profile_unigram": profile_unigram,
                "profile_bigram": profile_bigram,
                "profile_context": profile_context,
            }

    def score(
        self,
        text: str,
        profile: str,
        *,
        lexical: bool,
        scope: str,
        order: int,
        prior: float,
    ) -> float:
        data = self.data[lexical]
        global_unigram: Counter[str] = data["global_unigram"]  # type: ignore[assignment]
        global_bigram: Counter[tuple[str, str]] = data["global_bigram"]  # type: ignore[assignment]
        global_context: Counter[str] = data["global_context"]  # type: ignore[assignment]
        profile_unigram: dict[str, Counter[str]] = data["profile_unigram"]  # type: ignore[assignment]
        profile_bigram: dict[str, Counter[tuple[str, str]]] = data["profile_bigram"]  # type: ignore[assignment]
        profile_context: dict[str, Counter[str]] = data["profile_context"]  # type: ignore[assignment]

        tokens = lexical_tokens(text) if lexical else raw_tokens(text)
        sequence = ["<s>", *tokens, "</s>"]
        vocabulary = len(global_unigram) + 1
        global_total = sum(global_unigram.values())

        def global_uni(word: str) -> float:
            return (global_unigram[word] + 0.1) / (global_total + 0.1 * vocabulary)

        def global_bi(previous: str, word: str) -> float:
            return (global_bigram[(previous, word)] + 10.0 * global_uni(word)) / (
                global_context[previous] + 10.0
            )

        if scope == "global":
            if order == 1:
                probabilities = [global_uni(word) for word in sequence[1:]]
            else:
                probabilities = [
                    global_bi(previous, word)
                    for previous, word in zip(sequence, sequence[1:])
                ]
        elif order == 1:
            counts = profile_unigram.get(profile, Counter())
            total = sum(counts.values())
            probabilities = [
                (counts[word] + prior * global_uni(word)) / (total + prior)
                for word in sequence[1:]
            ]
        else:
            bigrams = profile_bigram.get(profile, Counter())
            contexts = profile_context.get(profile, Counter())
            probabilities = [
                (bigrams[(previous, word)] + prior * global_bi(previous, word))
                / (contexts[previous] + prior)
                for previous, word in zip(sequence, sequence[1:])
            ]
        return -sum(math.log(max(1e-300, value)) for value in probabilities) / max(
            1, len(probabilities)
        )
