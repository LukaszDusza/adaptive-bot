# logic/position_manager.py
from position import Position


class PositionManager:
    def __init__(self, config):
        self.config = config
        self.active_position: Position | None = None
        self.events = []

    def _log_event(self, timestamp, event_type, details):
        self.events.append({"timestamp": timestamp, "event": event_type, "details": details})

    def process_candle(self, current_candle, analysis, capital):
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

        # Aktualizacja mechanik BE i TSL
        self._update_mechanics(current_candle)

        # Logika wyjścia sygnałem z modelu
        votes_short = self._count_votes(analysis, prediction_target=0)
        if pos.strategy == 'long':
            if votes_short >= 2:
                pos.opposing_signal_count += 1
            else:
                pos.opposing_signal_count = 0
            if pos.opposing_signal_count >= self.config.EXIT_SIGNAL_PERSISTENCE:
                exit_reason = "Model Exit Signal"

        # Logika wyjścia przez SL/TP
        if not exit_reason:
            if pos.strategy == 'long':
                if current_candle['low'] <= pos.current_sl_price:
                    exit_reason = "Break-Even" if pos.is_be else ("Trailing Stop" if pos.is_trailing else "Stop Loss")
                elif not pos.is_trailing and current_candle['high'] >= pos.tp_price:
                    exit_reason = "Take Profit"

        if exit_reason:
            sl_tp_price_map = {"Stop Loss": pos.current_sl_price, "Trailing Stop": pos.current_sl_price,
                               "Break-Even": pos.current_sl_price, "Take Profit": pos.tp_price}
            exit_price = current_candle['close'] if "Model Exit" in exit_reason else sl_tp_price_map.get(exit_reason,
                                                                                                         current_candle[
                                                                                                             'close'])
            pnl = (exit_price - pos.entry_price) * pos.size

            return {
                'entry_date': pos.entry_date, 'exit_date': current_candle.name,
                'entry_price': pos.entry_price, 'exit_price': exit_price,
                'size': pos.size, 'pnl_usd': pnl, 'exit_reason': exit_reason,
                'strategy': pos.strategy, 'conf_momentum': pos.conf_momentum,
                'conf_reversion': pos.conf_reversion, 'conf_pa': pos.conf_pa
            }
        return None

    def _check_for_new_entry(self, current_candle, analysis, capital):
        if self.active_position: return None

        votes_long = self._count_votes(analysis, prediction_target=1)
        strategy_to_open = 'long' if votes_long >= self.config.ENTRY_VOTES else None

        if strategy_to_open:
            entry_price = analysis['current_price']
            stop_loss_distance = analysis['atr_value_5m'] * self.config.ATR_MULTIPLIER
            sl_price = entry_price - stop_loss_distance
            tp_price = entry_price + (abs(entry_price - sl_price) * self.config.RRR)

            position_value = capital * self.config.RISK_PERCENT * self.config.LEVERAGE
            position_size = position_value / entry_price if entry_price > 0 else 0

            tp_distance = abs(tp_price - entry_price)
            be_trigger = self.config.BREAKEVEN_TRIGGER_PERCENT
            be_trigger_price = entry_price + (tp_distance * be_trigger) if be_trigger > 0 else 0
            be_sl_price = entry_price + (
                        self.config.TRADE_COST_USD / position_size) if position_size > 0 else entry_price

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
        return None

    def _update_mechanics(self, current_candle):
        pos = self.active_position
        if pos.strategy == 'long':
            if self.config.TRAILING_SL_TRIGGER_R > 0 and not pos.is_trailing and current_candle[
                'high'] >= pos.trailing_trigger_price:
                pos.is_trailing = True
                self._log_event(current_candle.name, 'trailing_sl_activated', {'trade_entry_date': pos.entry_date})
            elif self.config.BREAKEVEN_TRIGGER_PERCENT > 0 and not pos.is_be and current_candle[
                'high'] >= pos.breakeven_trigger_price:
                pos.current_sl_price = pos.breakeven_sl_price;
                pos.is_be = True
                self._log_event(current_candle.name, 'breakeven_activated', {'trade_entry_date': pos.entry_date})
            if pos.is_trailing:
                new_sl = current_candle['close'] - (current_candle['ATRr_14_5m'] * self.config.TRAILING_SL_DISTANCE_ATR)
                if new_sl > pos.current_sl_price: pos.current_sl_price = new_sl

    def _count_votes(self, analysis, prediction_target):
        votes = 0
        min_conf_map = {'momentum': self.config.MIN_CONF_MOMENTUM, 'reversion': self.config.MIN_CONF_REVERSION,
                        'pa': self.config.MIN_CONF_PA}
        for expert, opinion in analysis['expert_opinions'].items():
            if opinion['confidence'] >= min_conf_map[expert] and opinion['prediction'] == prediction_target:
                votes += 1
        return votes