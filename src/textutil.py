"""Deterministic text helpers shared by validation, scoring, and ATS checks.

No AI and no I/O. Everything here is pure so it can be tested directly:

* term matching with synonyms and sane word boundaries ("Java" != "JavaScript")
* numeric "atoms" (numbers, %, $, counts, number words) for metric checks
* ownership-verb detection driven by the lexicon in config.py
* small date / slug / similarity helpers
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import config

# ---------------------------------------------------------------------------
# Basic normalization
# ---------------------------------------------------------------------------

_PUNCT_MAP = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-",
    " ": " ", " ": " ", " ": " ", " ": " ", "​": "",
    "…": "...", "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
}
_PUNCT_RE = re.compile("|".join(map(re.escape, _PUNCT_MAP)))


def plain(text: str) -> str:
    """Unify quotes, dashes, ligatures, and odd spaces; collapse whitespace."""
    text = _PUNCT_RE.sub(lambda m: _PUNCT_MAP[m.group(0)], text or "")
    return re.sub(r"\s+", " ", text).strip()


def normalize(text: str) -> str:
    """`plain()` plus lower-casing. Used for loose comparisons."""
    return plain(text).lower()


def squash(text: str) -> str:
    """Lower-case and drop everything except letters and digits.

    Lets us check containment without caring about line wraps, bullets,
    punctuation, or spacing (used for quote and PDF-extraction checks).
    """
    return re.sub(r"[^a-z0-9]+", "", normalize(text))


def words(text: str) -> list[str]:
    """Lower-cased word tokens with surrounding punctuation stripped."""
    return re.findall(r"[a-z0-9]+(?:['./+#&-][a-z0-9+#]+)*[+#]*", normalize(text))


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def similarity(a: str, b: str) -> float:
    """Word-level similarity in [0, 1]."""
    return difflib.SequenceMatcher(None, words(a), words(b)).ratio()


def same_text(a: str, b: str) -> bool:
    """True when two strings differ only in case, punctuation, or spacing."""
    return squash(a) == squash(b)


def slug(text: str, max_len: int = 40) -> str:
    """Filesystem-safe slug ("Goldman Sachs & Co." -> "Goldman_Sachs_Co")."""
    ascii_text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", ascii_text).strip("_")
    return (cleaned[:max_len].rstrip("_")) or "unknown"


# ---------------------------------------------------------------------------
# Synonyms and term matching
# ---------------------------------------------------------------------------


def _clean_term(term: str) -> str:
    return re.sub(r"\s+", " ", (term or "").strip().lower())


@lru_cache(maxsize=1)
def _variant_to_canon() -> dict[str, str]:
    table: dict[str, str] = {}
    for canonical, variants in config.SYNONYMS.items():
        target = _clean_term(canonical)
        for form in (canonical, *variants):
            cleaned = _clean_term(form)
            table[cleaned] = target
            table.setdefault(cleaned.replace("-", " "), target)   # "scikit learn" == "scikit-learn"
    return table


def canon(term: str) -> str:
    """Canonical lower-case form of a skill / keyword (synonym-aware)."""
    cleaned = _clean_term(term)
    table = _variant_to_canon()
    if cleaned in table:
        return table[cleaned]
    # "full stack" and "full-stack" are the same term.
    dehyphenated = re.sub(r"\s+", " ", cleaned.replace("-", " ")).strip()
    return table.get(dehyphenated, dehyphenated)


def surface_forms(term: str) -> list[str]:
    """Every spelling that should count as a mention of `term`."""
    canonical = canon(term)
    forms = {term.strip(), canonical}
    for key, variants in config.SYNONYMS.items():
        if _clean_term(key) == canonical:
            forms.add(key)
            forms.update(variants)
    return sorted((f for f in forms if f), key=len, reverse=True)


@lru_cache(maxsize=1)
def _case_sensitive_lookup() -> dict[str, str]:
    return {t.lower(): t for t in config.CASE_SENSITIVE_TERMS}


@lru_cache(maxsize=4096)
def _term_regex(form: str) -> re.Pattern[str]:
    """Regex for one surface form with word boundaries that respect + # . in names."""
    case_sensitive = False
    literal = form
    lowered = form.lower()
    if lowered in _case_sensitive_lookup():
        literal, case_sensitive = _case_sensitive_lookup()[lowered], True
    elif len(form) <= 2 and form.isalpha():
        literal, case_sensitive = form.upper(), True          # "r" -> "R", "ml" -> "ML"
    elif form.isupper() and len(form) <= 4 and form.isalpha():
        case_sensitive = True                                 # acronyms: "NLP", "LLM"
    body = re.escape(literal)
    body = re.sub(r"(\\ |\\-)+", r"[\\s\\-]+", body)          # "full-stack" == "full stack"
    if literal[-1].isalpha() and not literal.lower().endswith("s") and len(literal) > 3:
        body += "s?"                                          # light plural handling
    pattern = r"(?<![A-Za-z0-9])" + body + r"(?![A-Za-z0-9+#])"
    return re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)


def term_spans(text: str, term: str) -> list[tuple[int, int]]:
    """Character spans where `term` (or a synonym) is mentioned in `text`."""
    haystack = plain(text)
    spans: list[tuple[int, int]] = []
    for form in surface_forms(term):
        spans.extend(m.span() for m in _term_regex(form).finditer(haystack))
    return spans


def contains_term(text: str, term: str) -> bool:
    return bool(term_spans(text, term))


def count_term(text: str, term: str) -> int:
    """Number of distinct (non-overlapping) mentions of `term` in `text`."""
    spans = sorted(set(term_spans(text, term)))
    count, last_end = 0, -1
    for start, end in spans:
        if start >= last_end:
            count += 1
            last_end = end
    return count


def find_terms(text: str, vocabulary: Iterable[str]) -> dict[str, str]:
    """Vocabulary terms mentioned in `text`: {canonical term: matched surface text}.

    A term whose match sits wholly inside a longer matched term is dropped, so
    "GitHub Pages" is not also reported as "GitHub".
    """
    haystack = plain(text)
    matches: list[tuple[int, int, str]] = []
    seen_terms: set[str] = set()
    for term in vocabulary:
        canonical = canon(term)
        if not canonical or canonical in seen_terms:
            continue
        seen_terms.add(canonical)
        for start, end in term_spans(haystack, term):
            matches.append((start, end, canonical))
    found: dict[str, str] = {}
    for start, end, canonical in matches:
        swallowed = any(
            (s <= start and end <= e) and (e - s) > (end - start) and other != canonical
            for s, e, other in matches
        )
        if not swallowed and canonical not in found:
            found[canonical] = haystack[start:end]
    return found


# ---------------------------------------------------------------------------
# Numeric atoms (metrics)
# ---------------------------------------------------------------------------

_NUMBER_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "dozen": 12, "dozens": 12, "hundred": 100,
    "hundreds": 100, "thousand": 1000, "thousands": 1000, "million": 1_000_000,
    "millions": 1_000_000, "billion": 1_000_000_000, "billions": 1_000_000_000,
}
_ORDINAL_WORDS = {
    "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9, "tenth": 10,
}
_MULTIPLIER_WORDS = {
    "double": 2, "doubled": 2, "doubles": 2, "doubling": 2, "twice": 2, "twofold": 2,
    "triple": 3, "tripled": 3, "triples": 3, "tripling": 3, "threefold": 3,
    "quadruple": 4, "quadrupled": 4, "tenfold": 10,
    "half": 0.5, "halved": 0.5, "halving": 0.5,
}
_SUFFIX_MULTIPLIER = {"k": 1e3, "m": 1e6, "mm": 1e6, "b": 1e9, "bn": 1e9}

_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9.])"
    r"(?P<cur>[$€£])?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<suf>\s?%|\s?percent\b|\s?percentage\b|(?:st|nd|rd|th)\b|(?:k|m|mm|b|bn)\b|x\b)?"
    r"(?P<plus>\+)?"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
# Mixed letter+digit tokens such as "3D", "Web3", "H1B" are compared verbatim.
_ALNUM_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]+(?![A-Za-z0-9])")
_NUMERIC_SUFFIX_TOKEN_RE = re.compile(r"^\d+(?:st|nd|rd|th|k|m|mm|b|bn|x)$", re.IGNORECASE)


@dataclass(frozen=True)
class Atom:
    """One number-like claim. `unit` is "", "%", "$", "x", "ord", or "tok"."""

    value: str          # normalized number as a string, or the token for "tok"
    unit: str = ""
    more: bool = False  # "20+", "over 20", "more than 20"
    raw: str = ""

    def key(self) -> tuple[str, str]:
        return (self.value, self.unit)


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.6g}"


_TIME_SPAN_RE = re.compile(r"[\s-]*(?:day|week|month|year|semester|quarter|summer|season)s?\b", re.IGNORECASE)


def _has_marker_before(text: str, start: int, markers: Iterable[str], end: int | None = None) -> bool:
    """True when a "more than" marker precedes the number at `start`.

    "over 3 weeks" / "over two years" describe a duration, not "more than", so
    "over" followed by a time span is not treated as a marker.
    """
    window = text[max(0, start - 22):start].lower()
    for marker in markers:
        if re.search(r"\b" + re.escape(marker) + r"\s+(?:the\s+|a\s+)?$", window):
            if marker == "over" and end is not None and _TIME_SPAN_RE.match(text[end:]):
                continue
            return True
    return False


def numeric_atoms(text: str) -> list[Atom]:
    """Every number, %, $, count, ordinal, multiplier, or number word in `text`."""
    source = plain(text)
    atoms: list[Atom] = []
    for m in _NUMBER_RE.finditer(source):
        value = float(m.group("num").replace(",", ""))
        suffix = (m.group("suf") or "").strip().lower()
        unit = ""
        if m.group("cur"):
            unit = "$"
        if suffix in ("%", "percent", "percentage"):
            unit = "%"
        elif suffix in ("st", "nd", "rd", "th"):
            unit = "ord"
        elif suffix == "x":
            unit = "x"
        elif suffix in _SUFFIX_MULTIPLIER:
            value *= _SUFFIX_MULTIPLIER[suffix]
        more = bool(m.group("plus")) or _has_marker_before(source, m.start(), config.STRICT_MORE_MARKERS, m.end())
        atoms.append(Atom(_fmt(value), unit, more, m.group(0)))
    for m in _ALNUM_TOKEN_RE.finditer(source):
        token = m.group(0)
        if not _NUMERIC_SUFFIX_TOKEN_RE.match(token):
            atoms.append(Atom(token.lower(), "tok", False, token))
    for m in re.finditer(r"[A-Za-z]+", source):
        word = m.group(0).lower()
        if word in _NUMBER_WORDS:
            more = _has_marker_before(source, m.start(), config.STRICT_MORE_MARKERS, m.end())
            atoms.append(Atom(_fmt(_NUMBER_WORDS[word]), "", more, m.group(0)))
        elif word in _ORDINAL_WORDS:
            atoms.append(Atom(_fmt(_ORDINAL_WORDS[word]), "ord", False, m.group(0)))
        elif word in _MULTIPLIER_WORDS:
            atoms.append(Atom(_fmt(_MULTIPLIER_WORDS[word]), "x", False, m.group(0)))
    return atoms


def unsupported_atoms(text: str, evidence_texts: Iterable[str]) -> list[Atom]:
    """Atoms in `text` that no evidence text backs up.

    A "more than" atom ("20+", "over 20") needs evidence that also says more.
    """
    evidence: dict[tuple[str, str], bool] = {}
    for item in evidence_texts:
        for atom in numeric_atoms(item):
            evidence[atom.key()] = evidence.get(atom.key(), False) or atom.more
    missing = []
    for atom in numeric_atoms(text):
        if atom.key() not in evidence or (atom.more and not evidence[atom.key()]):
            missing.append(atom)
    return missing


# ---------------------------------------------------------------------------
# Ownership verbs
# ---------------------------------------------------------------------------


def _regular_forms(lemma: str) -> list[str]:
    base = lemma
    if base.endswith("e"):
        past, gerund = base + "d", base[:-1] + "ing"
    elif base.endswith("y") and len(base) > 2 and base[-2] not in "aeiou":
        past, gerund = base[:-1] + "ied", base + "ing"
    else:
        past, gerund = base + "ed", base + "ing"
    if base.endswith(("s", "x", "z", "ch", "sh")):
        third = base + "es"
    elif base.endswith("y") and len(base) > 2 and base[-2] not in "aeiou":
        third = base[:-1] + "ies"
    else:
        third = base + "s"
    return [base, third, past, gerund]


def verb_forms(lemma: str) -> list[str]:
    return list(config.VERB_FORMS.get(lemma) or _regular_forms(lemma))


@lru_cache(maxsize=1)
def _ownership_patterns() -> list[tuple[str, str, int, re.Pattern[str], tuple[str, ...]]]:
    """(lemma, form, tier, regex, excluded-next-words) for every lexicon verb form."""
    out = []
    for tier_name, lemmas in config.OWNERSHIP_VERBS.items():
        tier = config.OWNERSHIP_TIERS[tier_name]
        for lemma in lemmas:
            for form in verb_forms(lemma):
                body = re.escape(form).replace(r"\ ", r"\s+")
                regex = re.compile(r"(?<![A-Za-z-])" + body + r"(?![A-Za-z])", re.IGNORECASE)
                out.append((lemma, form, tier, regex, tuple(config.VERB_FORM_EXCLUSIONS.get(form, ()))))
    for tier_name, nouns in config.OWNERSHIP_NOUNS.items():
        tier = config.OWNERSHIP_TIERS[tier_name]
        for noun in nouns:
            body = re.escape(noun).replace(r"\ ", r"\s+").replace(r"\-", r"[\s\-]?")
            regex = re.compile(r"(?<![A-Za-z-])" + body + r"s?(?![A-Za-z])", re.IGNORECASE)
            out.append(("noun:" + noun, noun, tier, regex, ()))
    return out


def ownership_terms(text: str) -> dict[str, int]:
    """Ownership-lexicon lemmas used in `text` mapped to their tier (1-4)."""
    source = plain(text)
    found: dict[str, int] = {}
    for lemma, _form, tier, regex, excluded in _ownership_patterns():
        for m in regex.finditer(source):
            following = re.match(r"\s+([A-Za-z]+)", source[m.end():])
            if excluded and following and following.group(1).lower() in excluded:
                continue
            found[lemma] = max(tier, found.get(lemma, 0))
            break
    # "team lead" should not also count as the bare verb "lead".
    if any(k.startswith("noun:") and k.endswith(" lead") for k in found) and "lead" in found:
        without_nouns = re.sub(r"(?i)\b(team|tech|project)\s+lead\b", " ", source)
        if not any(r.search(without_nouns) for l, _f, _t, r, _e in _ownership_patterns() if l == "lead"):
            del found["lead"]
    return found


def is_ambiguous_ownership(lemma: str) -> bool:
    return lemma in config.AMBIGUOUS_OWNERSHIP_VERBS


def downgrade_qualifiers(text: str) -> set[str]:
    """Helper-role verbs present in `text` ("assisted", "helped", ...)."""
    return {lemma for lemma in ownership_terms(text) if lemma in config.DOWNGRADE_QUALIFIERS}


def tier_name(tier: int) -> str:
    return {1: "member", 2: "contributor", 3: "lead", 4: "founder/owner"}.get(tier, "?")


# ---------------------------------------------------------------------------
# Word-list matching (scope, complexity, result words)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2048)
def _phrase_regex(phrase: str) -> re.Pattern[str]:
    body = re.escape(phrase.lower())
    body = re.sub(r"(\\ |\\-)+", r"[\\s\\-]?", body)        # "end-to-end" == "end to end" == "endtoend"
    return re.compile(r"(?<![A-Za-z0-9])" + body + r"(?![A-Za-z0-9])", re.IGNORECASE)


def phrases_in(text: str, phrases: Iterable[str]) -> set[str]:
    """Which of `phrases` appear in `text` (whole words, hyphen/space tolerant)."""
    source = plain(text)
    return {p for p in phrases if _phrase_regex(p).search(source)}


def stems_in(text: str, stems: Iterable[str]) -> set[str]:
    """Which verb stems appear in any inflection ("improve" matches "improved")."""
    source = plain(text)
    found = set()
    for stem in stems:
        forms = "|".join(re.escape(f) for f in _regular_forms(stem))
        if re.search(r"(?<![A-Za-z])(?:" + forms + r")(?![A-Za-z])", source, re.IGNORECASE):
            found.add(stem)
    return found


_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by", "at", "from",
    "as", "into", "across", "through", "that", "which", "this", "these", "those", "it",
    "its", "their", "our", "my", "is", "are", "was", "were", "be", "been", "while", "using",
    "via", "within", "per", "over", "under", "up", "out", "than", "both", "each",
}


def content_words(text: str) -> set[str]:
    return {w for w in words(text) if w not in _STOPWORDS and len(w) > 1}


def new_proper_nouns(text: str, evidence_texts: Iterable[str]) -> list[str]:
    """Capitalized / tech-shaped tokens in `text` that the evidence never mentions.

    The first word of the bullet is skipped (it is capitalized by position).
    """
    evidence = squash(" ".join(evidence_texts))
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9+#./&'-]*", plain(text))
    out: list[str] = []
    for index, token in enumerate(tokens):
        token = token.rstrip(".,;:'")
        if index == 0 or len(token) < 2:
            continue
        shaped = token[0].isupper() or any(c.isupper() for c in token[1:]) or any(c in "+#" for c in token)
        if not shaped:
            continue
        if squash(token) and squash(token) not in evidence and token not in out:
            out.append(token)
    return out


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def parse_ym(value: str, *, end: bool = False) -> tuple[int, int]:
    """("2024-10") -> (2024, 10). "present" sorts last; "" sorts first.

    A bare year counts as December when it is an end date and January otherwise.
    """
    value = (value or "").strip().lower()
    if value == "present":
        return (9999, 12)
    m = re.match(r"^(\d{4})(?:-(\d{2}))?$", value)
    if not m:
        return (0, 0)
    month = int(m.group(2)) if m.group(2) else (12 if end else 1)
    return (int(m.group(1)), month)


def format_date(value: str) -> str:
    value = (value or "").strip()
    if value.lower() == "present":
        return "Present"
    m = re.match(r"^(\d{4})-(\d{2})$", value)
    if m:
        return f"{_MONTHS[int(m.group(2)) - 1]} {m.group(1)}"
    return value


def format_date_range(start: str, end: str) -> str:
    """"2024-11".."2024-12" -> "Nov – Dec 2024"; "2024-10".."present" -> "Oct 2024 – Present"."""
    start, end = (start or "").strip(), (end or "").strip()
    if not start and not end:
        return ""
    if not end or start == end:
        return format_date(start or end)
    if not start:
        return format_date(end)
    s, e = re.match(r"^(\d{4})-(\d{2})$", start), re.match(r"^(\d{4})-(\d{2})$", end)
    if s and e and s.group(1) == e.group(1):
        return f"{_MONTHS[int(s.group(2)) - 1]} – {_MONTHS[int(e.group(2)) - 1]} {e.group(1)}"
    return f"{format_date(start)} – {format_date(end)}"


def years_in(start: str, end: str) -> list[str]:
    """Years covered by a date range (so a bullet may mention them)."""
    s, e = parse_ym(start), parse_ym(end, end=True)
    if s[0] == 0 and e[0] == 0:
        return []
    first = s[0] or e[0]
    last = e[0] if 0 < e[0] < 9999 else first
    return [str(y) for y in range(first, max(first, last) + 1)]
