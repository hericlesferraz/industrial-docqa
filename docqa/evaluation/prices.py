"""Dated price table: measured tokens times list price give the cost the report shows.

Prices are US$ per million tokens of the Gemini API paid tier, read from the provider's pricing page on the date
below; images count as input tokens and thinking tokens are billed as output. A model missing from the table has no
cost in the report, which says so instead of guessing.
"""

from dataclasses import dataclass

PRICES_AS_OF = "2026-10-30"
PRICES_SOURCE = "Gemini API pricing page, paid tier (standard, not batch)"


@dataclass(frozen=True)
class Price:
    """List price of one model.

    Attributes:
        input_per_m: US$ per million input tokens.
        output_per_m: US$ per million output tokens, thinking included.
    """

    input_per_m: float
    output_per_m: float


PRICES: dict[str, Price] = {
    "gemini-3.5-flash-lite": Price(0.30, 2.50),
    "gemini-3.8-flash": Price(0.825, 4.125),
    "gemini-embedding-2": Price(0.20, 0.0),
}


def cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Price some tokens of a model.

    Args:
        model: Model name as requested.
        input_tokens: Input tokens.
        output_tokens: Output tokens, thinking included.

    Returns:
        The cost in US$, or None for a model missing from the table.
    """
    price = PRICES.get(model)
    if price is None:
        return None
    return (input_tokens * price.input_per_m + output_tokens * price.output_per_m) / 1_000_000
