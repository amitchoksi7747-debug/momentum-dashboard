from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class PriceProvider(ABC):
    """Common interface every price source implements, so strategy code never
    knows which provider a bar came from."""

    name: str = "base"

    @abstractmethod
    def get_daily(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        """Return long-format DataFrame: Symbol, Date, Open, High, Low, Close, Volume, Source.

        `start`/`end` are ISO date strings (end inclusive where the source allows).
        Symbols use the canonical NSE:X form used throughout this repo.
        """
        raise NotImplementedError
