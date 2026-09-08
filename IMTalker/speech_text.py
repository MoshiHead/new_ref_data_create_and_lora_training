"""speech_text.py — spoken-form text normalization for injected search summaries.

Everything the search pipeline injects into PersonaPlex as a `<ref>` block is
text the model is expected to READ ALOUD. PersonaPlex is a speech model: it has
no text normalizer of its own, so whatever symbols survive into the `<ref>`
block are what it tries to pronounce. In practice that produced two distinct
failure modes in the old pipeline's logs:

  * symbol soup -- markdown (`**`, `##`, `|`), citation brackets, stray
    parentheses and pipes read out as literal noises or silently derailed the
    sentence;
  * digits and currency signs -- "$4,059.41" was spoken as a mangled mixture of
    "dollar sign", digit-by-digit reads, and skipped decimals, and "€325"
    routinely lost the currency entirely.

This module converts a compressed summary into the form a person would say:

    "$325"          -> "Three hundred twenty-five dollars."
    "€325"          -> "Three hundred twenty-five euros."
    "$4,059.41"     -> "Four thousand fifty-nine dollars and forty-one cents."
    "$1.2 billion"  -> "One point two billion dollars."
    "up 25%"        -> "Up twenty-five percent."
    "3.14"          -> "Three point one four."

Design rules, in priority order:

1. NEVER lose the fact. The whole point of a search turn is the number, so a
   pattern that cannot be normalized confidently is left alone rather than
   dropped -- a slightly awkward read beats a missing figure.
2. Currency before plain numbers. "$325" must become "dollars", not "three
   hundred twenty-five" with the sign deleted, so the currency pass runs first
   and consumes its number.
3. Pure functions, no model, no I/O. This runs inline on the search thread
   between compression and injection, where the budget is microseconds.

Nothing here is specific to the search path -- `normalize_for_speech` is safe
to call on any string that is about to be spoken.
"""
from __future__ import annotations

import re
import unicodedata

__all__ = [
    "normalize_for_speech",
    "number_to_words",
    "SPOKEN_STYLE_RULES",
]


# ── Cardinal numbers ────────────────────────────────────────────────────────

_ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_TENS = (
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
)
# Short scale (US/UK modern), which is what English-language web results use.
_SCALES = (
    (10 ** 12, "trillion"),
    (10 ** 9, "billion"),
    (10 ** 6, "million"),
    (10 ** 3, "thousand"),
)


def _under_hundred(n: int) -> str:
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    # Hyphenated, matching ordinary written English ("twenty-five"). The
    # hyphen is deliberately kept by the symbol filter below for exactly this.
    return _TENS[tens] if ones == 0 else f"{_TENS[tens]}-{_ONES[ones]}"


def _under_thousand(n: int) -> str:
    if n < 100:
        return _under_hundred(n)
    hundreds, rest = divmod(n, 100)
    head = f"{_ONES[hundreds]} hundred"
    return head if rest == 0 else f"{head} {_under_hundred(rest)}"


def _integer_to_words(n: int) -> str:
    """Cardinal form of a non-negative integer.

    No "and" is inserted before the tens ("four thousand fifty-nine", not
    "four thousand and fifty-nine"): the "and" slot is reserved for the
    currency minor unit ("... dollars and forty-one cents"), where its
    presence is what tells a listener a second quantity is starting."""
    if n == 0:
        return "zero"
    parts: list[str] = []
    for value, name in _SCALES:
        if n >= value:
            count, n = divmod(n, value)
            parts.append(f"{_under_thousand(count)} {name}")
    if n:
        parts.append(_under_thousand(n))
    return " ".join(parts)


def number_to_words(text: str) -> str:
    """Spoken form of one numeric literal, given as the raw matched string.

    Accepts an optional sign, thousands separators, and a decimal part:
    "-1,204.5" -> "minus one thousand two hundred four point five". A decimal
    tail is read digit by digit, which is how prices, versions and rates are
    actually said aloud ("point four one", never "point forty-one")."""
    cleaned = text.strip().replace(",", "").replace(" ", "")
    negative = cleaned.startswith("-")
    cleaned = cleaned.lstrip("+-")
    if not cleaned or not cleaned.replace(".", "", 1).isdigit():
        return text
    if "." in cleaned:
        whole, _, frac = cleaned.partition(".")
        whole_words = _integer_to_words(int(whole or 0))
        frac_words = " ".join(_ONES[int(d)] for d in frac)
        words = f"{whole_words} point {frac_words}".strip() if frac else whole_words
    else:
        words = _integer_to_words(int(cleaned))
    return f"minus {words}" if negative else words


# ── Currency ────────────────────────────────────────────────────────────────
#
# (major singular, major plural, minor singular, minor plural). The minor unit
# is only ever used for an exactly-two-digit decimal tail -- "$4,059.41" is
# forty-one cents, but "$1.2 billion" is one point two billion dollars, and
# conflating those two shapes is the classic money-normalizer bug.

_CURRENCIES = {
    "$": ("dollar", "dollars", "cent", "cents"),
    "us$": ("dollar", "dollars", "cent", "cents"),
    "usd": ("dollar", "dollars", "cent", "cents"),
    "€": ("euro", "euros", "cent", "cents"),          # €
    "eur": ("euro", "euros", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),        # £
    "gbp": ("pound", "pounds", "penny", "pence"),
    "¥": ("yen", "yen", "sen", "sen"),                 # ¥
    "jpy": ("yen", "yen", "sen", "sen"),
    "₹": ("rupee", "rupees", "paisa", "paise"),        # ₹
    "inr": ("rupee", "rupees", "paisa", "paise"),
    "₩": ("won", "won", "jeon", "jeon"),               # ₩
    "₽": ("ruble", "rubles", "kopek", "kopeks"),       # ₽
    "₺": ("lira", "lira", "kurus", "kurus"),           # ₺
    "৳": ("taka", "taka", "poisha", "poisha"),         # ৳
    "bdt": ("taka", "taka", "poisha", "poisha"),
}

# Magnitude words that may follow the digits ("$1.2 billion"). When one is
# present the decimal tail is a fraction of the magnitude, not a minor unit.
_MAGNITUDE_WORDS = {
    "k": "thousand", "thousand": "thousand",
    "m": "million", "mm": "million", "mn": "million", "million": "million",
    "b": "billion", "bn": "billion", "billion": "billion",
    "t": "trillion", "tn": "trillion", "trillion": "trillion",
}

_CURRENCY_SYMBOLS = "".join(
    sym for sym in _CURRENCIES if len(sym) == 1 and not sym.isalpha()
)
_MAGNITUDE_ALT = "|".join(sorted(_MAGNITUDE_WORDS, key=len, reverse=True))

# Symbol-first form: "$325", "€4,059.41", "£1.2 billion".
_CURRENCY_PREFIX_RE = re.compile(
    rf"(?P<sym>[{re.escape(_CURRENCY_SYMBOLS)}]|\bUS\$)\s*"
    rf"(?P<num>\d[\d,]*(?:\.\d+)?)"
    rf"(?:\s*(?P<mag>{_MAGNITUDE_ALT})\b)?",
    re.IGNORECASE,
)
# Code-suffix form: "325 USD", "4,059.41 EUR".
_CURRENCY_SUFFIX_RE = re.compile(
    rf"(?P<num>\d[\d,]*(?:\.\d+)?)\s*"
    rf"(?:(?P<mag>{_MAGNITUDE_ALT})\s*)?"
    rf"\b(?P<code>USD|EUR|GBP|JPY|INR|BDT)\b",
    re.IGNORECASE,
)


def _spoken_money(raw_num: str, magnitude: str, spec: tuple[str, str, str, str]) -> str:
    major_sg, major_pl, minor_sg, minor_pl = spec
    digits = raw_num.replace(",", "")
    whole, _, frac = digits.partition(".")
    whole_int = int(whole or 0)

    if magnitude:
        # "$1.2 billion": the decimal belongs to the magnitude, so read the
        # whole literal as one number and append the scale word.
        return f"{number_to_words(digits)} {_MAGNITUDE_WORDS[magnitude.lower()]} {major_pl}"

    major_words = _integer_to_words(whole_int)
    major_unit = major_sg if whole_int == 1 else major_pl

    if len(frac) == 2:
        minor_int = int(frac)
        if minor_int == 0:
            return f"{major_words} {major_unit}"
        minor_unit = minor_sg if minor_int == 1 else minor_pl
        minor_words = f"{_integer_to_words(minor_int)} {minor_unit}"
        # "$0.75" is seventy-five cents, not "zero dollars and seventy-five
        # cents" -- naming a zero major unit is how nobody says a sub-unit
        # price out loud.
        if whole_int == 0:
            return minor_words
        return f"{major_words} {major_unit} and {minor_words}"
    if frac:
        # An unusual tail length (".5", ".415") is not a minor unit; read it
        # as a decimal so no digits are invented or dropped.
        return f"{number_to_words(digits)} {major_pl}"
    return f"{major_words} {major_unit}"


def _expand_currency(text: str) -> str:
    def prefix_sub(m: re.Match) -> str:
        spec = _CURRENCIES.get(m.group("sym").lower())
        if spec is None:
            return m.group(0)
        return _spoken_money(m.group("num"), m.group("mag") or "", spec)

    def suffix_sub(m: re.Match) -> str:
        spec = _CURRENCIES.get(m.group("code").lower())
        if spec is None:
            return m.group(0)
        return _spoken_money(m.group("num"), m.group("mag") or "", spec)

    text = _CURRENCY_PREFIX_RE.sub(prefix_sub, text)
    return _CURRENCY_SUFFIX_RE.sub(suffix_sub, text)


# ── Other numeric shapes ────────────────────────────────────────────────────

_PERCENT_RE = re.compile(r"(?P<num>\d[\d,]*(?:\.\d+)?)\s*%")
_MAGNITUDE_PLAIN_RE = re.compile(
    rf"\b(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<mag>{_MAGNITUDE_ALT})\b", re.IGNORECASE
)
_TIME_RE = re.compile(r"\b(?P<h>[01]?\d|2[0-3]):(?P<m>[0-5]\d)\s*(?P<ap>[ap]\.?m\.?)?", re.IGNORECASE)
_ORDINAL_RE = re.compile(r"\b(?P<num>\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
_PLAIN_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

_ORDINAL_WORDS = {
    "one": "first", "two": "second", "three": "third", "five": "fifth",
    "eight": "eighth", "nine": "ninth", "twelve": "twelfth",
}


def _ordinal_words(n: int) -> str:
    """Ordinal form built by inflecting only the LAST word of the cardinal:
    "twenty-one" -> "twenty-first", "forty" -> "fortieth". Splitting on the
    final hyphen or space is what keeps "one hundred first" from becoming
    "oneth hundred first"."""
    words = _integer_to_words(n)
    for sep in ("-", " "):
        head, found, tail = words.rpartition(sep)
        if found:
            return f"{head}{sep}{_inflect_ordinal(tail)}"
    return _inflect_ordinal(words)


def _inflect_ordinal(word: str) -> str:
    if word in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[word]
    if word.endswith("y"):
        return word[:-1] + "ieth"
    return word + "th"


def _spoken_clock(m: re.Match) -> str:
    hour = _integer_to_words(int(m.group("h")))
    minute = m.group("m")
    body = f"{hour} o'clock" if minute == "00" else f"{hour} {_integer_to_words(int(minute))}"
    meridiem = m.group("ap") or ""
    if meridiem:
        body += " a m" if meridiem.lower().startswith("a") else " p m"
    return body


def _expand_numbers(text: str) -> str:
    text = _PERCENT_RE.sub(lambda m: f"{number_to_words(m.group('num'))} percent", text)
    text = _MAGNITUDE_PLAIN_RE.sub(
        lambda m: f"{number_to_words(m.group('num'))} {_MAGNITUDE_WORDS[m.group('mag').lower()]}",
        text,
    )
    text = _TIME_RE.sub(_spoken_clock, text)
    text = _ORDINAL_RE.sub(lambda m: _ordinal_words(int(m.group("num"))), text)
    return _PLAIN_NUMBER_RE.sub(lambda m: number_to_words(m.group(0)), text)


# ── Symbol and whitespace cleanup ───────────────────────────────────────────
#
# Runs LAST, after every numeric pass has had its chance to consume the
# symbols it needs ($, %, :). What survives to here is genuinely decorative.

_SYMBOL_WORDS = [
    # Abbreviations first, and each one swallows its own trailing period. A
    # pattern ending in `\.?\b` cannot do that -- there is no word boundary
    # after a period followed by a space -- so "approx." became
    # "approximately." mid-sentence and the capitalizer then treated the next
    # word as a new sentence.
    (re.compile(r"\bvs\.?(?=\s|$)", re.IGNORECASE), "versus"),
    (re.compile(r"\bapprox\.?(?=\s|$)", re.IGNORECASE), "approximately"),
    (re.compile(r"\bestd?\.?(?=\s|$)", re.IGNORECASE), "estimated"),
    (re.compile(r"\be\.g\.\s*", re.IGNORECASE), "for example "),
    (re.compile(r"\bi\.e\.\s*", re.IGNORECASE), "that is "),
    (re.compile(r"\s*&\s*"), " and "),
    (re.compile(r"\s*\+\s*"), " plus "),
    # "and/or" is the one common slash that means "or"; every other slash in
    # price/market text is a rate ("$/oz", "km/h"), which is read "per".
    (re.compile(r"\band\s*/\s*or\b", re.IGNORECASE), "and or"),
    (re.compile(r"(?<=\w)\s*/\s*(?=\w)"), " per "),
    (re.compile(r"\s*=\s*"), " equals "),
    (re.compile(r"\s*@\s*"), " at "),
    (re.compile(r"°\s*[CF]?"), " degrees "),
    # Numeric ranges were already turned into "X to Y" by the pre-pass, so
    # any dash still standing here is an aside. It reads as a comma pause,
    # not a spurious "to".
    (re.compile(r"\s*[–—]\s*"), ", "),
]

# Run before any numeric expansion, while both sides are still digits.
_RANGE_RE = re.compile(r"(?<=\d)\s*[-–—]\s*(?=[\d$€£¥₹₩₽₺৳])")

# Markdown / citation / bracket furniture, removed outright.
_MARKUP_RE = re.compile(r"\[[^\]]*\]\([^)]*\)|[*_`~#|<>\[\]{}]+")
_PARENS_RE = re.compile(r"[()]")
_QUOTES_RE = re.compile(r"[\"“”‘’«»]")
# Anything left that is not a letter, digit, space, or sentence punctuation.
# Digits are still allowed through here because _expand_numbers has already
# run: what remains is something it deliberately declined to touch, and a
# digit read aloud is far better than a deleted one.
_RESIDUAL_RE = re.compile(r"[^\w\s.,!?'\-]", re.UNICODE)
# Same as _RESIDUAL_RE but keeps currency and percent signs. Used when numbers
# are left as digits: stripping "$" off "$309.32" would silently change the
# fact, which is worse than keeping one symbol the audio decoder handles fine.
_RESIDUAL_KEEP_UNITS_RE = re.compile(r"[^\w\s.,!?'\-$€£¥₹₩₽₺৳%]", re.UNICODE)
_SPACES_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,!?])")
_REPEAT_PUNCT_RE = re.compile(r"([.,!?])\1+")
_ORPHAN_HYPHEN_RE = re.compile(r"(?<![\w])-+|-+(?![\w])")


def _strip_symbols(text: str, keep_units: bool = False) -> str:
    text = _MARKUP_RE.sub(" ", text)
    text = _PARENS_RE.sub(" ", text)
    text = _QUOTES_RE.sub(" ", text)
    for pattern, replacement in _SYMBOL_WORDS:
        text = pattern.sub(replacement, text)
    text = (_RESIDUAL_KEEP_UNITS_RE if keep_units else _RESIDUAL_RE).sub(" ", text)
    text = _ORPHAN_HYPHEN_RE.sub(" ", text)
    text = _SPACES_RE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _REPEAT_PUNCT_RE.sub(r"\1", text)
    return text.strip()


_SENTENCE_START_RE = re.compile(r"(^|[.!?]\s+)([a-z])")


def _capitalize_sentences(text: str) -> str:
    return _SENTENCE_START_RE.sub(lambda m: m.group(1) + m.group(2).upper(), text)


# ── Public entry point ──────────────────────────────────────────────────────

def normalize_for_speech(
    text: str,
    ensure_terminal_period: bool = True,
    spell_numbers: bool = False,
) -> str:
    """Clean a summary for speech, optionally spelling numbers out in words.

    `spell_numbers` defaults to FALSE, and that default is load-bearing rather
    than a preference. PersonaPlex reads "$309.32" aloud as "three hundred nine
    dollars and thirty-two cents" by itself -- the conversion belongs to its
    audio decoder, which does it reliably. Handing it the words instead forces
    the model to re-encode them, and it drops magnitudes: "three hundred nine
    dollars and thirty-two cents" came back as "$39.32", and a euro figure came
    back with a dollar sign. Digits survive because the model only has to copy
    them.

    Symbol and markdown stripping runs either way. That part never changes a
    value, so it is safe on every injection.

    Order matters and is not arbitrary: currency must claim its digits before
    the plain-number pass sees them, and the symbol filter must run last so it
    only ever deletes decoration the numeric passes already declined.

    Returns "" for empty input. Never raises -- this sits directly on the
    injection path, and a normalizer that throws would cost the turn its
    answer entirely; on any unexpected failure the original text is returned
    unchanged, which is still speakable, just less cleanly."""
    if not text or not text.strip():
        return ""
    try:
        # NFKC folds full-width digits and ligatures into their plain ASCII
        # equivalents so the numeric patterns below can actually see them.
        out = unicodedata.normalize("NFKC", str(text))
        out = _RANGE_RE.sub(" to ", out)
        if spell_numbers:
            out = _expand_currency(out)
            out = _expand_numbers(out)
        out = _strip_symbols(out, keep_units=not spell_numbers)
        out = _capitalize_sentences(out)
        if ensure_terminal_period and out and out[-1] not in ".!?":
            out += "."
        return out
    except Exception as e:  # pragma: no cover - defensive only
        print(f"[speech_text] normalization failed, using raw text: {e!r}", flush=True)
        return str(text).strip()


# Instruction block appended to the compressor prompt so the model produces
# spoken form directly. `normalize_for_speech` still runs afterwards -- the
# prompt improves the odds, the normalizer is what guarantees the result.
PLAIN_TEXT_RULES = (
    "- Plain words and digits only: no markdown, no brackets, no quotation "
    "marks, no citation markers. Keep the currency or percent sign that belongs "
    "to a number, and keep the number itself in digits -- write \"$325\", not "
    "\"325\" and not \"three hundred twenty-five dollars\". The speech model "
    "reads digits aloud correctly and mis-reads spelled-out numbers.\n"
)

# Only appended when --spoken_form_numbers is on. See normalize_for_speech for
# why spelling numbers into the injected text makes the spoken answer worse.
SPOKEN_STYLE_RULES = (
    "- Write the sentence exactly as it should be SPOKEN ALOUD. Use plain words "
    "only: no symbols, no markdown, no brackets, no quotation marks, no "
    "currency signs, no percent signs.\n"
    "- Write every number in words, not digits. Say \"Three hundred twenty-five "
    "dollars\", never \"$325\". Say \"Three hundred twenty-five euros\", never "
    "\"€325\". Say \"twenty-five percent\", never \"25%\".\n"
)
