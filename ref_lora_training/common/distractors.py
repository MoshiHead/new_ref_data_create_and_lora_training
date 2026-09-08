"""distractors.py — fixed pool of "no search needed" follow-up turns, used to
build ExampleType.SCOPE_NEGATIVE episodes.

These are deliberately the SAME question shapes search_helpers.rule_route_explain
already classifies as "no search" (procedural / advisory / conceptual /
chitchat -- see IMTalker/search_helpers.py's _RULE_STATIC_FORMS and
_RULE_CHITCHAT_RE). That is the point: in production, a turn like this
following a grounded <ref> turn arrives with NO new <ref> block (the router
correctly decided it needs no lookup), so the model must answer it from
general knowledge -- and specifically must NOT keep repeating the previous
turn's injected number, which is exactly the contamination bug observed in
the project's conversation log (turns #6/#7 answering unrelated garbled
follow-ups with "310.36." and "32." after a Tesla stock <ref>).

Every reply here is intentionally generic and number-free, so
ref_format.validate_episode's leak check has something unambiguous to compare
against.
"""
from __future__ import annotations

DISTRACTOR_TURNS: list[dict] = [
    {
        "user_text": "Is it a good idea to diversify my investments?",
        "assistant_text": (
            "Generally, yes. Spreading money across different assets rather than "
            "putting it all in one place tends to reduce risk, though the right mix "
            "depends on your own goals and how much risk you're comfortable with."
        ),
    },
    {
        "user_text": "How do I open a brokerage account?",
        "assistant_text": (
            "You'd typically pick a brokerage, provide some identification and tax "
            "details, fund the account with a bank transfer, and then you're able to "
            "place trades. Most brokers walk you through it step by step online."
        ),
    },
    {
        "user_text": "What's the difference between a stock and a bond?",
        "assistant_text": (
            "A stock is a small ownership stake in a company, so its value moves with "
            "how the company performs. A bond is more like a loan you make to a company "
            "or government, which pays you back with interest over time."
        ),
    },
    {
        "user_text": "Thanks, that's helpful.",
        "assistant_text": "You're welcome. Let me know if there's anything else you'd like to go over.",
    },
    {
        "user_text": "Do you like music?",
        "assistant_text": (
            "I don't really have personal tastes, but I'm happy to chat about it if "
            "you'd like to tell me what you're into."
        ),
    },
    {
        "user_text": "Why do interest rates affect the stock market?",
        "assistant_text": (
            "Higher interest rates make borrowing more expensive and safer assets like "
            "savings accounts more attractive, so investors often shift money away from "
            "stocks, which tends to push prices down."
        ),
    },
    {
        "user_text": "Should I pay off debt before investing?",
        "assistant_text": (
            "It often makes sense to clear high-interest debt first, since the interest "
            "you're paying can easily outweigh typical investment returns. Lower-interest "
            "debt is more of a personal judgment call."
        ),
    },
    {
        "user_text": "Can you tell me a bit about how mutual funds work?",
        "assistant_text": (
            "A mutual fund pools money from many investors and a manager uses it to buy "
            "a mix of stocks, bonds, or other assets, so you get diversification without "
            "having to pick each investment yourself."
        ),
    },
    {
        "user_text": "What should I consider before buying cryptocurrency?",
        "assistant_text": (
            "Worth thinking about how much volatility you can tolerate, whether you "
            "understand the specific coin or project, and only investing money you "
            "could afford to lose given how sharply prices can swing."
        ),
    },
    {
        "user_text": "How's it going?",
        "assistant_text": "I'm doing well, thanks for asking. What can I help you with?",
    },
    {
        "user_text": "What are the risks of day trading?",
        "assistant_text": (
            "Day trading tends to carry high transaction costs, requires constant "
            "attention, and most people who try it underperform a simple buy-and-hold "
            "approach over time, so it's considered fairly high risk."
        ),
    },
    {
        "user_text": "Explain what inflation means.",
        "assistant_text": (
            "Inflation is the general rise in prices over time, which means each unit "
            "of currency buys a little less than it used to."
        ),
    },
]
