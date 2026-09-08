"""generation_prompts.py — the prompt template handed to an LLM to convert a
raw (question, context, answer) QA row into one grounded training episode.

The instructions here are deliberately built from the SAME rules
`ContextCompressor.compress()` uses in production
(`IMTalker/search_helpers.py`), via the shared `PLAIN_TEXT_RULES` import, so
the training data teaches the model to consume exactly the style of sentence
the live compressor actually produces -- not a cleaner or richer version of
it that the model will never see at inference time.
"""
from __future__ import annotations

from .ref_format import PLAIN_TEXT_RULES

CONVERSION_SYSTEM_PROMPT = """You are building training data for a real-time voice assistant named Robert. \
Robert answers spoken questions out loud. Sometimes, right before Robert answers, a short \
"grounding fact" is silently inserted into his context so he can answer with current information \
he wouldn't otherwise know. You are given a factual question/answer pair from a QA dataset. Turn it \
into three things a voice-assistant training example needs. Always respond with a single JSON object \
and nothing else -- no markdown fences, no commentary."""

CONVERSION_USER_TEMPLATE = """Source question: {question}
Source answer / supporting fact: {answer}
{context_block}
Produce a JSON object with exactly these fields:

"spoken_question": Rewrite the source question as something a person would casually ASK OUT LOUD to \
a voice assistant, phrased as if the fact is current / recent (use words like "right now", "today", \
"currently", "latest" where natural). Keep it short, one sentence, natural spoken English. Do not \
just copy the source question verbatim if it reads like a written exam question.

"ref_fact": Compress the source answer into ONE short sentence (20 words or fewer) stating the specific \
fact (the number, name, date, or figure) that answers the question, written so it can be read aloud. \
Do not summarize the whole source passage -- state only the one fact the question asked for.
Rules for this field:
{plain_text_rules}- Never mention "the passage", "the document", "the context", or any source -- state \
the fact directly as if it is simply known.
- Do not answer conversationally or add commentary -- this is a grounding note, not a reply.

"spoken_answer": Write Robert's actual spoken reply to the "spoken_question", naturally using the fact \
from "ref_fact" as if he simply knows it. Rules:
- One to three natural spoken sentences, first person confident tone, no meta-commentary like \
"according to my information" or "based on the search".
- Never use markdown, brackets, or the words "context", "reference", "passage", or "document".
- Never output any of the literal characters "<", ">", or the words "ref" or "lookup" as markup -- \
those are internal system tags and must never appear in spoken text.
- The reply must actually state the specific fact from "ref_fact" (the same number/name/date), just \
phrased conversationally -- do not vaguely gesture at it without saying it.

CRITICAL FORMATTING RULE: this must be valid JSON. ALL THREE field values must be wrapped in double \
quotes -- including "ref_fact" and "spoken_answer", not just "spoken_question". If a value itself \
contains a double quote, escape it as \\". Do not add line breaks inside a value. Do not write anything \
before or after the JSON object -- no preamble, no closing remarks.

Example of the exact shape required (write your own content, keep this structure):
{{"spoken_question": "How is Apple's revenue looking this quarter?", "ref_fact": "Apple reported quarterly revenue of $94.9 billion.", "spoken_answer": "Apple brought in about ninety four point nine billion dollars in revenue this quarter."}}

Respond with only the JSON object, matching that shape exactly."""

EXPECTED_FIELDS = ["spoken_question", "ref_fact", "spoken_answer"]


def build_conversion_prompt(question: str, answer: str, context: str = "") -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for one QA-row -> episode conversion call."""
    context_block = f"Source context excerpt: {context.strip()[:500]}\n" if context and context.strip() else ""
    user = CONVERSION_USER_TEMPLATE.format(
        question=question.strip(),
        answer=answer.strip(),
        context_block=context_block,
        plain_text_rules=PLAIN_TEXT_RULES,
    )
    return CONVERSION_SYSTEM_PROMPT, user
