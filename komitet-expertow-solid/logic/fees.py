# logic/fees.py
"""
Moduł zarządzający opłatami i poślizgami dla pozycji.
Kompatybilny z trybem backtest i live trading.
"""

from typing import Dict, Any


class FeeCalculator:
    """Kalkulator opłat i poślizgów dla pozycji."""
    
    def __init__(self, config):
        self.config = config
    
    def get_slippage_bps(self, exit_type: str) -> float:
        """
        Zwraca poślizg w bps (1/10000) dla danego typu wyjścia.
        
        Args:
            exit_type: 'stop' lub 'tp'
        
        Returns:
            Poślizg jako ułamek dziesiętny (np. 0.0005 dla 5 bps)
        """
        if exit_type == 'stop':
            return float(getattr(self.config, "SLIPPAGE_BPS_STOP", 5.0)) / 10_000.0
        else:  # 'tp'
            return float(getattr(self.config, "SLIPPAGE_BPS_TP", 2.0)) / 10_000.0
    
    def apply_slippage(self, exit_reason: str, strategy: str, raw_price: float, 
                      candle: Dict[str, Any]) -> float:
        """
        Oblicza cenę wyjścia po zastosowaniu poślizgu.
        Poślizg jest zawsze niekorzystny i ograniczony do high/low świecy.
        
        Args:
            exit_reason: Powód wyjścia ("Stop Loss", "Take Profit", etc.)
            strategy: 'long' lub 'short'
            raw_price: Surowa cena wyjścia przed poślizgiem
            candle: Dane świecy z polami 'high' i 'low'
        
        Returns:
            Cena po zastosowaniu poślizgu
        """
        high = float(candle['high'])
        low = float(candle['low'])
        
        # Dla stop loss, trailing stop i break-even
        if exit_reason in ("Stop Loss", "Trailing Stop", "Break-Even"):
            slippage = self.get_slippage_bps('stop')
            
            if strategy == 'long':
                # Long: poślizg w dół (niekorzystny)
                slipped_price = raw_price * (1.0 - slippage)
                return max(low, slipped_price)  # Nie może być poniżej low
            else:  # short
                # Short: poślizg w górę (niekorzystny)
                slipped_price = raw_price * (1.0 + slippage)
                return min(high, slipped_price)  # Nie może być powyżej high
        
        # Dla take profit
        elif exit_reason == "Take Profit":
            slippage = self.get_slippage_bps('tp')
            
            if strategy == 'long':
                # Long: poślizg w dół (niekorzystny)
                slipped_price = raw_price * (1.0 - slippage)
                return max(low, min(high, slipped_price))
            else:  # short
                # Short: poślizg w górę (niekorzystny)
                slipped_price = raw_price * (1.0 + slippage)
                return max(low, min(high, slipped_price))
        
        # Dla model exit i innych przypadków - bez poślizgu
        return raw_price
    
    def calculate_pnl(self, strategy: str, entry_price: float, exit_price: float, 
                     size: float) -> float:
        """
        Oblicza PnL pozycji.
        
        Args:
            strategy: 'long' lub 'short'
            entry_price: Cena wejścia
            exit_price: Cena wyjścia
            size: Rozmiar pozycji
        
        Returns:
            PnL w USD
        """
        if strategy == 'long':
            return (exit_price - entry_price) * size
        else:  # short
            return (entry_price - exit_price) * size
    
    def calculate_exchange_fees(self, notional: float, fee_bps: float) -> float:
        """
        Oblicza opłaty giełdy na podstawie nominału i opłaty w bps.
        
        Args:
            notional: Wartość nominalna transakcji (cena * rozmiar)
            fee_bps: Opłata w bps (basis points)
        
        Returns:
            Opłata w USD
        """
        return float(notional) * float(fee_bps) / 10_000.0


def get_fee_calculator(config) -> FeeCalculator:
    """Factory function do tworzenia kalkulatora opłat."""
    return FeeCalculator(config)