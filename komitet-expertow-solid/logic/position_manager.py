# logic/position_manager.py
from .position import Position
from .fees import get_fee_calculator
from typing import Dict, Any, Optional, Tuple


class PositionManager:
    def __init__(self, config):
        self.config = config
        self.active_position: Position | None = None
        self.events = []
        self.fee_calculator = get_fee_calculator(config)

    def _log_event(self, timestamp, event_type, details):
        self.events.append({"timestamp": timestamp, "event": event_type, "details": details})


    # ====== BE classification helper ======
    def _is_be_price(self, pos: Position) -> bool:
        """Czy bieżący SL leży na entry (w granicach tolerancji)?"""
        tol = float(getattr(self.config, "BE_TOL", 1e-9))
        return abs(pos.current_sl_price - pos.entry_price) <= tol

    def process_candle(self, current_candle, analysis, capital) -> Tuple[Optional[str], Optional[Any]]:
        """
        Przetwarza świecę i podejmuje decyzje o pozycjach.
        
        Returns:
            Tuple[action, details] gdzie action to 'OPEN', 'CLOSE' lub None
        """
        # Zastosuj oczekujące zmiany na pozycji
        self._apply_pending_changes()
        
        # Sprawdź czy zamknąć istniejącą pozycję
        if self.active_position:
            closed_trade = self._manage_active_position(current_candle, analysis)
            if closed_trade:
                self.active_position = None
                return 'CLOSE', closed_trade
        
        # Sprawdź czy otworzyć nową pozycję
        new_pos_details = self._check_for_new_entry(current_candle, analysis, capital)
        if new_pos_details:
            self.active_position = Position(**new_pos_details)
            return 'OPEN', self.active_position
        
        return None, None

    def _apply_pending_changes(self):
        """Stosuje oczekujące zmiany na aktywnej pozycji (BE i TSL)."""
        if not self.active_position:
            return
        
        self._apply_pending_breakeven()
        self._apply_pending_trailing_stop()

    def _apply_pending_breakeven(self):
        """Stosuje oczekujący break-even."""
        pos = self.active_position
        if not getattr(pos, "_pending_be", False):
            return
        
        # Ustaw SL na poziom break-even
        if pos.strategy == 'long':
            if pos.breakeven_sl_price > pos.current_sl_price:
                pos.current_sl_price = pos.breakeven_sl_price
        else:  # short
            if pos.breakeven_sl_price < pos.current_sl_price:
                pos.current_sl_price = pos.breakeven_sl_price
        
        pos.is_be = True
        pos._pending_be = False

    def _apply_pending_trailing_stop(self):
        """Stosuje oczekujący trailing stop."""
        pos = self.active_position
        if not hasattr(pos, "_pending_tsl_sl"):
            return
        
        candidate_sl = pos._pending_tsl_sl
        if candidate_sl is not None:
            if pos.strategy == 'long':
                if candidate_sl > pos.current_sl_price:
                    pos.current_sl_price = candidate_sl
            else:  # short
                if candidate_sl < pos.current_sl_price:
                    pos.current_sl_price = candidate_sl
            
            pos._pending_tsl_sl = None

    def _manage_active_position(self, current_candle, analysis) -> Optional[Dict[str, Any]]:
        """Zarządza aktywną pozycją i zwraca dane transakcji wyjścia jeśli pozycja zostanie zamknięta."""
        pos = self.active_position
        
        # Aktualizacja mechanik BE i TSL
        self._update_mechanics(current_candle)
        
        # Sprawdź sygnał wyjścia z modelu
        exit_reason = self._check_model_exit_signal(pos, analysis)
        
        # Sprawdź wyjście przez SL/TP (priorytet: SL przed TP)
        if not exit_reason:
            exit_reason = self._check_stop_take_exit(pos, current_candle)
        
        if exit_reason:
            return self._create_exit_trade(pos, exit_reason, current_candle)
        
        return None

    def _create_exit_trade(self, pos: Position, exit_reason: str, current_candle) -> Dict[str, Any]:
        """Tworzy dane transakcji wyjścia z pozycji."""
        raw_exit_price = self._get_raw_exit_price(exit_reason, pos, current_candle)
        exit_price = self.fee_calculator.apply_slippage(
            exit_reason, pos.strategy, raw_exit_price, current_candle
        )
        pnl = self.fee_calculator.calculate_pnl(
            pos.strategy, pos.entry_price, exit_price, pos.size
        )
        
        return {
            'entry_date': pos.entry_date,
            'exit_date': current_candle.name,
            'entry_price': pos.entry_price,
            'exit_price': exit_price,
            'size': pos.size,
            'pnl_usd': pnl,
            'exit_reason': exit_reason,
            'strategy': pos.strategy,
            'conf_momentum': pos.conf_momentum,
            'conf_reversion': pos.conf_reversion,
            'conf_pa': pos.conf_pa
        }

    def _get_raw_exit_price(self, exit_reason: str, pos: Position, current_candle) -> float:
        """Zwraca surową cenę wyjścia przed zastosowaniem poślizgu."""
        if "Model Exit" in exit_reason:
            return current_candle['close']
        
        exit_price_map = {
            "Stop Loss": pos.current_sl_price,
            "Trailing Stop": pos.current_sl_price,
            "Break-Even": pos.current_sl_price,
            "Take Profit": pos.tp_price
        }
        return exit_price_map.get(exit_reason, current_candle['close'])

    def _check_model_exit_signal(self, pos: Position, analysis) -> Optional[str]:
        """Sprawdza czy wystąpił sygnał wyjścia z modelu."""
        votes_short = self._count_votes(analysis, prediction_target=0)
        if pos.strategy == 'long':
            if votes_short >= 2:
                pos.opposing_signal_count += 1
            else:
                pos.opposing_signal_count = 0
            if pos.opposing_signal_count >= self.config.EXIT_SIGNAL_PERSISTENCE:
                return "Model Exit Signal"
        return None

    def _check_stop_take_exit(self, pos: Position, current_candle) -> Optional[str]:
        """Sprawdza czy pozycja powinna zostać zamknięta przez SL/TP."""
        if pos.strategy == 'long':
            if current_candle['low'] <= pos.current_sl_price:
                return self._classify_stop_reason(pos)
            elif current_candle['high'] >= pos.tp_price:
                return "Take Profit"
        else:  # short
            if current_candle['high'] >= pos.current_sl_price:
                return self._classify_stop_reason(pos)
            elif current_candle['low'] <= pos.tp_price:
                return "Take Profit"
        return None

    def _classify_stop_reason(self, pos: Position) -> str:
        """Klasyfikuje powód zamknięcia przez stop."""
        if self._is_be_price(pos):
            return "Break-Even"
        elif pos.is_trailing:
            return "Trailing Stop"
        else:
            return "Stop Loss"

    def _check_for_new_entry(self, current_candle, analysis, capital) -> Optional[Dict[str, Any]]:
        """Sprawdza czy można otworzyć nową pozycję i zwraca jej parametry."""
        if not analysis or self.active_position:
            return None

        strategy_to_open = self._determine_entry_strategy(analysis)
        if not strategy_to_open:
            return None

        return self._calculate_position_parameters(current_candle, analysis, capital, strategy_to_open)

    def _determine_entry_strategy(self, analysis) -> Optional[str]:
        """Określa strategię wejścia na podstawie głosów ekspertów."""
        votes_long = self._count_votes(analysis, prediction_target=1)
        return 'long' if votes_long >= self.config.ENTRY_VOTES else None

    def _calculate_position_parameters(self, current_candle, analysis, capital, strategy: str) -> Dict[str, Any]:
        """Oblicza wszystkie parametry nowej pozycji."""
        entry_price = analysis['current_price']
        sl_price, tp_price = self._calculate_sl_tp_prices(entry_price, analysis, strategy)
        position_size = self._calculate_position_size(entry_price, sl_price, capital)
        
        be_trigger_price, be_sl_price = self._calculate_breakeven_params(entry_price, tp_price)
        tsl_trigger_price = self._calculate_trailing_params(entry_price, analysis)
        
        return {
            'strategy': strategy,
            'entry_date': current_candle.name,
            'entry_price': entry_price,
            'size': position_size,
            'current_sl_price': sl_price,
            'tp_price': tp_price,
            'breakeven_trigger_price': be_trigger_price,
            'breakeven_sl_price': be_sl_price,
            'trailing_trigger_price': tsl_trigger_price,
            'conf_momentum': analysis['expert_opinions']['momentum']['confidence'],
            'conf_reversion': analysis['expert_opinions']['reversion']['confidence'],
            'conf_pa': analysis['expert_opinions']['pa']['confidence']
        }

    def _calculate_sl_tp_prices(self, entry_price: float, analysis, strategy: str) -> Tuple[float, float]:
        """Oblicza ceny stop loss i take profit."""
        stop_loss_distance = analysis['atr_value_5m'] * self.config.ATR_MULTIPLIER
        
        if strategy == 'long':
            sl_price = entry_price - stop_loss_distance
        else:  # short
            sl_price = entry_price + stop_loss_distance
        
        tp_distance = abs(entry_price - sl_price) * self.config.RRR
        tp_price = entry_price + tp_distance if strategy == 'long' else entry_price - tp_distance
        
        return sl_price, tp_price

    def _calculate_position_size(self, entry_price: float, sl_price: float, capital: float) -> float:
        """Oblicza rozmiar pozycji na podstawie zarządzania ryzykiem."""
        sl_distance = abs(entry_price - sl_price)
        risk_usd = capital * self.config.RISK_PERCENT
        return (risk_usd / sl_distance) if sl_distance > 0 else 0

    def _calculate_breakeven_params(self, entry_price: float, tp_price: float) -> Tuple[float, float]:
        """Oblicza parametry break-even."""
        tp_distance = abs(tp_price - entry_price)
        be_trigger = self.config.BREAKEVEN_TRIGGER_PERCENT
        be_trigger_price = entry_price + (tp_distance * be_trigger) if be_trigger > 0 else 0
        be_sl_price = entry_price  # BE = entry (realne 0)
        return be_trigger_price, be_sl_price

    def _calculate_trailing_params(self, entry_price: float, analysis) -> float:
        """Oblicza parametry trailing stop."""
        stop_loss_distance = analysis['atr_value_5m'] * self.config.ATR_MULTIPLIER
        tsl_trigger = self.config.TRAILING_SL_TRIGGER_R
        return entry_price + (stop_loss_distance * tsl_trigger) if tsl_trigger > 0 else 0

    def _update_mechanics(self, current_candle):
        pos = self.active_position
        if pos.strategy == 'long':
            # 1) Trailing – aktywacja po R-multiple (stan), bez natychmiastowego podnoszenia SL
            if self.config.TRAILING_SL_TRIGGER_R > 0 and not pos.is_trailing and current_candle['high'] >= pos.trailing_trigger_price:
                pos.is_trailing = True
                self._log_event(current_candle.name, 'trailing_sl_activated', {'trade_entry_date': pos.entry_date})

            # 2) BE – tylko TRIGGER na tej świecy; zastosujemy na początku następnej
            elif self.config.BREAKEVEN_TRIGGER_PERCENT > 0 and not pos.is_be and current_candle['high'] >= pos.breakeven_trigger_price:
                pos._pending_be = True
                self._log_event(current_candle.name, 'breakeven_activated', {'trade_entry_date': pos.entry_date})

            # 3) Trailing – zapisujemy kandydata SL do zastosowania na N+1
            if pos.is_trailing:
                atr_here = current_candle['ATRr_14_5m']
                new_sl_candidate = current_candle['close'] - (atr_here * self.config.TRAILING_SL_DISTANCE_ATR)
                best = getattr(pos, "_pending_tsl_sl", None)
                if (best is None) or (new_sl_candidate > best):
                    pos._pending_tsl_sl = new_sl_candidate

    def _count_votes(self, analysis, prediction_target):
        votes = 0
        min_conf_map = {
            'momentum': self.config.MIN_CONF_MOMENTUM,
            'reversion': self.config.MIN_CONF_REVERSION,
            'pa': self.config.MIN_CONF_PA
        }
        for expert, opinion in analysis['expert_opinions'].items():
            if opinion['confidence'] >= min_conf_map[expert] and opinion['prediction'] == prediction_target:
                votes += 1
        return votes
