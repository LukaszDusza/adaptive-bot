# logic/position_manager.py
from .position import Position


class PositionManager:
    def __init__(self, config):
        self.config = config
        self.active_position: Position | None = None
        self.events = []

    def _log_event(self, timestamp, event_type, details):
        self.events.append({"timestamp": timestamp, "event": event_type, "details": details})

    # ====== Slippage helper ======
    def _slip_bps(self, kind: str) -> float:
        """Zwraca bps (1/10000) poślizgu dla danego typu wyjścia."""
        if kind == 'stop':
            return float(getattr(self.config, "SLIPPAGE_BPS_STOP", 5.0)) / 10_000.0  # 0.05%
        else:
            return float(getattr(self.config, "SLIPPAGE_BPS_TP", 2.0)) / 10_000.0    # 0.02%

    def _apply_slippage(self, reason: str, pos: Position, raw_price: float, candle) -> float:
        """
        Cena wyjścia po poślizgu (z ograniczeniem do high/low świecy). Poślizg zawsze niekorzystny.
        """
        high = float(candle['high'])
        low = float(candle['low'])

        if reason in ("Stop Loss", "Trailing Stop", "Break-Even"):
            s = self._slip_bps('stop')
            if pos.strategy == 'long':
                slipped = raw_price * (1.0 - s)
                return max(low, slipped)
            else:
                slipped = raw_price * (1.0 + s)
                return min(high, slipped)

        if reason == "Take Profit":
            s = self._slip_bps('tp')
            if pos.strategy == 'long':
                slipped = raw_price * (1.0 - s)
                return max(low, min(high, slipped))
            else:
                slipped = raw_price * (1.0 + s)
                return max(low, min(high, slipped))

        # Model Exit / fallback – bez poślizgu
        return raw_price

    # ====== BE classification helper ======
    def _is_be_price(self, pos: Position) -> bool:
        """Czy bieżący SL leży na entry (w granicach tolerancji)?"""
        tol = float(getattr(self.config, "BE_TOL", 1e-9))
        return abs(pos.current_sl_price - pos.entry_price) <= tol

    def process_candle(self, current_candle, analysis, capital):
        # --- APPLY PENDING BE (apply-on-next-bar) ---
        if self.active_position and getattr(self.active_position, "_pending_be", False):
            pos = self.active_position
            # BE = SL na poziom breakeven_sl_price (u Ciebie: entry_price)
            if pos.strategy == 'long':
                if pos.breakeven_sl_price > pos.current_sl_price:
                    pos.current_sl_price = pos.breakeven_sl_price
            else:
                if pos.breakeven_sl_price < pos.current_sl_price:
                    pos.current_sl_price = pos.breakeven_sl_price
            pos.is_be = True
            pos._pending_be = False  # wyczyść flagę

        # --- APPLY PENDING TSL (apply-on-next-bar) ---
        if self.active_position and hasattr(self.active_position, "_pending_tsl_sl"):
            pos = self.active_position
            cand = pos._pending_tsl_sl
            if cand is not None:
                if pos.strategy == 'long':
                    if cand > pos.current_sl_price:
                        pos.current_sl_price = cand
                else:
                    if cand < pos.current_sl_price:
                        pos.current_sl_price = cand
                pos._pending_tsl_sl = None  # skonsumowane

        # Krok 1: Sprawdź, czy zamknąć istniejącą pozycję
        if self.active_position:
            closed_trade = self._manage_active_position(current_candle, analysis)
            if closed_trade:
                self.active_position = None
                return 'CLOSE', closed_trade

        # Krok 2: Sprawdź, czy otworzyć nową pozycję
        new_pos_details = self._check_for_new_entry(current_candle, analysis, capital)
        if new_pos_details:
            self.active_position = Position(**new_pos_details)
            return 'OPEN', self.active_position

        return None, None

    def _manage_active_position(self, current_candle, analysis):
        pos = self.active_position
        exit_reason = None

        # Aktualizacja mechanik BE i TSL (tu tylko TRIGGER/KANDYDAT)
        self._update_mechanics(current_candle)

        # Logika wyjścia sygnałem z modelu (dla longa – krótsze SYG)
        votes_short = self._count_votes(analysis, prediction_target=0)
        if pos.strategy == 'long':
            if votes_short >= 2:
                pos.opposing_signal_count += 1
            else:
                pos.opposing_signal_count = 0
            if pos.opposing_signal_count >= self.config.EXIT_SIGNAL_PERSISTENCE:
                exit_reason = "Model Exit Signal"

        # Logika wyjścia przez SL/TP (pesymistyczny priorytet: SL przed TP)
        if not exit_reason:
            if pos.strategy == 'long':
                if current_candle['low'] <= pos.current_sl_price:
                    # --- POPRAWKA: rozróżniaj BE vs TSL vs SL po POZIOMIE SL względem entry ---
                    if self._is_be_price(pos):
                        exit_reason = "Break-Even"
                    elif pos.is_trailing:
                        exit_reason = "Trailing Stop"
                    else:
                        exit_reason = "Stop Loss"
                elif current_candle['high'] >= pos.tp_price:
                    exit_reason = "Take Profit"
            else:  # short
                if current_candle['high'] >= pos.current_sl_price:
                    if self._is_be_price(pos):
                        exit_reason = "Break-Even"
                    elif pos.is_trailing:
                        exit_reason = "Trailing Stop"
                    else:
                        exit_reason = "Stop Loss"
                elif current_candle['low'] <= pos.tp_price:
                    exit_reason = "Take Profit"

        if exit_reason:
            # Cena surowa wg powodu
            if "Model Exit" in exit_reason:
                raw_exit_price = current_candle['close']
            else:
                sl_tp_price_map = {
                    "Stop Loss": pos.current_sl_price,
                    "Trailing Stop": pos.current_sl_price,
                    "Break-Even": pos.current_sl_price,
                    "Take Profit": pos.tp_price
                }
                raw_exit_price = sl_tp_price_map.get(exit_reason, current_candle['close'])

            # Zastosuj poślizg i policz PnL
            exit_price = self._apply_slippage(exit_reason, pos, raw_exit_price, current_candle)
            if pos.strategy == 'long':
                pnl = (exit_price - pos.entry_price) * pos.size
            else:
                pnl = (pos.entry_price - exit_price) * pos.size

            return {
                'entry_date': pos.entry_date, 'exit_date': current_candle.name,
                'entry_price': pos.entry_price, 'exit_price': exit_price,
                'size': pos.size, 'pnl_usd': pnl, 'exit_reason': exit_reason,
                'strategy': pos.strategy, 'conf_momentum': pos.conf_momentum,
                'conf_reversion': pos.conf_reversion, 'conf_pa': pos.conf_pa
            }

        return None

    def _check_for_new_entry(self, current_candle, analysis, capital):
        if not analysis or self.active_position:
            return None

        votes_long = self._count_votes(analysis, prediction_target=1)
        strategy_to_open = 'long' if votes_long >= self.config.ENTRY_VOTES else None

        if not strategy_to_open:
            return None

        entry_price = analysis['current_price']
        stop_loss_distance = analysis['atr_value_5m'] * self.config.ATR_MULTIPLIER
        sl_price = entry_price - stop_loss_distance
        tp_price = entry_price + (abs(entry_price - sl_price) * self.config.RRR)

        # risk-based sizing: ryzyko w $ = (|entry - SL|) * size
        sl_distance = abs(entry_price - sl_price)
        risk_usd = capital * self.config.RISK_PERCENT
        position_size = (risk_usd / sl_distance) if sl_distance > 0 else 0

        tp_distance = abs(tp_price - entry_price)

        be_trigger = self.config.BREAKEVEN_TRIGGER_PERCENT
        be_trigger_price = entry_price + (tp_distance * be_trigger) if be_trigger > 0 else 0
        be_sl_price = entry_price  # BE = entry (realne 0)

        tsl_trigger = self.config.TRAILING_SL_TRIGGER_R
        tsl_trigger_price = entry_price + (stop_loss_distance * tsl_trigger) if tsl_trigger > 0 else 0

        return {
            'strategy': strategy_to_open, 'entry_date': current_candle.name,
            'entry_price': entry_price, 'size': position_size,
            'current_sl_price': sl_price, 'tp_price': tp_price,
            'breakeven_trigger_price': be_trigger_price, 'breakeven_sl_price': be_sl_price,
            'trailing_trigger_price': tsl_trigger_price,
            'conf_momentum': analysis['expert_opinions']['momentum']['confidence'],
            'conf_reversion': analysis['expert_opinions']['reversion']['confidence'],
            'conf_pa': analysis['expert_opinions']['pa']['confidence']
        }

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
