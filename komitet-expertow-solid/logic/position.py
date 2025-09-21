# logic/position.py
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Position:
    strategy: str
    entry_date: datetime
    entry_price: float
    size: float

    current_sl_price: float
    tp_price: float

    is_be: bool = False
    is_trailing: bool = False

    breakeven_trigger_price: float = 0.0
    breakeven_sl_price: float = 0.0
    trailing_trigger_price: float = 0.0

    opposing_signal_count: int = 0

    # Dodane do raportowania
    conf_momentum: float = 0.0
    conf_reversion: float = 0.0
    conf_pa: float = 0.0