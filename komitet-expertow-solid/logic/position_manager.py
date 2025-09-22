# logic/position_manager.py
import logging
from .position import Position
from .fees import get_fee_calculator
from typing import Dict, Any, Optional, Tuple

# Setup logging
logger = logging.getLogger(__name__)


class PositionManager:
    def __init__(self, config):
        self.config = config
        self.active_position: Position | None = None
        self.events = []
        self.fee_calculator = get_fee_calculator(config)

    def _log_event(self, timestamp, event_type, details):
        self.events.append({"timestamp": timestamp, "event": event_type, "details": details})

    # ====== LIVE TRADING API METHODS ======
    
    def get_trading_signal(self, current_candle, analysis, capital) -> Dict[str, Any]:
        """
        API method for live trading - returns trading signal and instructions.
        
        Returns:
            dict with keys:
            - action: 'OPEN_LONG', 'OPEN_SHORT', 'CLOSE', 'HOLD'
            - confidence: ML confidence levels
            - entry_price: suggested entry price
            - stop_loss: suggested SL price
            - take_profit: suggested TP price
            - size: position size
            - instructions: list of position management instructions
        """
        # Log ML predictions at the start
        logger.info("=== ML PREDICTIONS ===")
        logger.info(f"Current price: {analysis['current_price']:.4f}")
        logger.info(f"ATR (5m): {analysis['atr_value_5m']:.6f}")
        
        expert_opinions = analysis['expert_opinions']
        for expert_name, opinion in expert_opinions.items():
            prediction_text = "BULLISH" if opinion['prediction'] == 1 else "BEARISH"
            logger.info(f"{expert_name.upper()}: {prediction_text} (confidence: {opinion['confidence']:.4f})")
        
        instructions = []
        
        # Check for position management instructions first
        if self.active_position:
            pos_instructions = self._get_position_instructions(current_candle, analysis)
            instructions.extend(pos_instructions)
            
            # Check for exit signal
            exit_reason = self._check_model_exit_signal(self.active_position, analysis)
            if not exit_reason:
                exit_reason = self._check_stop_take_exit(self.active_position, current_candle)
            
            if exit_reason:
                return {
                    'action': 'CLOSE',
                    'exit_reason': exit_reason,
                    'exit_price': self._get_raw_exit_price(exit_reason, self.active_position, current_candle),
                    'instructions': instructions,
                    'confidence': {
                        'momentum': analysis['expert_opinions']['momentum']['confidence'],
                        'reversion': analysis['expert_opinions']['reversion']['confidence'],
                        'pa': analysis['expert_opinions']['pa']['confidence']
                    }
                }
        
        # Check for new entry signal
        if not self.active_position:
            strategy = self._determine_entry_strategy(analysis)
            if strategy:
                position_params = self._calculate_position_parameters(current_candle, analysis, capital, strategy)
                
                return {
                    'action': f'OPEN_{strategy.upper()}',
                    'strategy': strategy,
                    'entry_price': position_params['entry_price'],
                    'stop_loss': position_params['current_sl_price'],
                    'take_profit': position_params['tp_price'],
                    'size': position_params['size'],
                    'breakeven_trigger': position_params['breakeven_trigger_price'],
                    'trailing_trigger': position_params['trailing_trigger_price'],
                    'instructions': instructions,
                    'confidence': {
                        'momentum': analysis['expert_opinions']['momentum']['confidence'],
                        'reversion': analysis['expert_opinions']['reversion']['confidence'],
                        'pa': analysis['expert_opinions']['pa']['confidence']
                    }
                }
        
        return {
            'action': 'HOLD',
            'instructions': instructions,
            'confidence': {
                'momentum': analysis['expert_opinions']['momentum']['confidence'],
                'reversion': analysis['expert_opinions']['reversion']['confidence'],
                'pa': analysis['expert_opinions']['pa']['confidence']
            }
        }
    
    def _get_position_instructions(self, current_candle, analysis) -> list:
        """
        Returns list of position management instructions for live trader.
        """
        instructions = []
        pos = self.active_position
        
        if not pos:
            return instructions
        
        # Check for break-even trigger
        if self.config.BREAKEVEN_TRIGGER_PERCENT > 0 and not pos.is_be:
            if pos.strategy == 'long':
                if current_candle['high'] >= pos.breakeven_trigger_price:
                    instructions.append({
                        'type': 'MOVE_SL_TO_BREAKEVEN',
                        'new_sl_price': pos.breakeven_sl_price,
                        'reason': 'Breakeven trigger reached'
                    })
            else:  # short
                if current_candle['low'] <= pos.breakeven_trigger_price:
                    instructions.append({
                        'type': 'MOVE_SL_TO_BREAKEVEN',
                        'new_sl_price': pos.breakeven_sl_price,
                        'reason': 'Breakeven trigger reached'
                    })
        
        # Check for trailing stop activation
        if self.config.TRAILING_SL_TRIGGER_R > 0 and not pos.is_trailing:
            if pos.strategy == 'long':
                if current_candle['high'] >= pos.trailing_trigger_price:
                    instructions.append({
                        'type': 'ACTIVATE_TRAILING_STOP',
                        'reason': 'Trailing stop trigger reached'
                    })
            else:  # short
                if current_candle['low'] <= pos.trailing_trigger_price:
                    instructions.append({
                        'type': 'ACTIVATE_TRAILING_STOP',
                        'reason': 'Trailing stop trigger reached'
                    })
        
        # Check for trailing stop update
        if pos.is_trailing:
            atr_here = current_candle['ATRr_14_5m']
            if pos.strategy == 'long':
                new_sl_candidate = current_candle['close'] - (atr_here * self.config.TRAILING_SL_DISTANCE_ATR)
                if new_sl_candidate > pos.current_sl_price:
                    instructions.append({
                        'type': 'UPDATE_TRAILING_STOP',
                        'new_sl_price': new_sl_candidate,
                        'reason': 'Trailing stop update'
                    })
            else:  # short
                new_sl_candidate = current_candle['close'] + (atr_here * self.config.TRAILING_SL_DISTANCE_ATR)
                if new_sl_candidate < pos.current_sl_price:
                    instructions.append({
                        'type': 'UPDATE_TRAILING_STOP',
                        'new_sl_price': new_sl_candidate,
                        'reason': 'Trailing stop update'
                    })
        
        return instructions
    
    def update_position_from_live_data(self, position_data: Dict[str, Any]):
        """
        Updates internal position state from live trading data.
        
        Args:
            position_data: Dict with keys like entry_price, size, current_sl_price, etc.
        """
        if not self.active_position:
            # Create position from live data
            self.active_position = Position(**position_data)
        else:
            # Update existing position
            for key, value in position_data.items():
                if hasattr(self.active_position, key):
                    setattr(self.active_position, key, value)
    
    def clear_position(self):
        """Clears the active position (called when position is closed in live trading)."""
        self.active_position = None
    
    def get_ml_predictions(self, analysis) -> Dict[str, Any]:
        """
        Returns ML model predictions and confidence levels separately from trading decisions.
        
        Returns:
            dict with model predictions, votes, and confidence levels
        """
        votes_long = self._count_votes(analysis, prediction_target=1)
        votes_short = self._count_votes(analysis, prediction_target=0)
        
        return {
            'votes_long': votes_long,
            'votes_short': votes_short,
            'total_experts': 3,
            'entry_threshold': self.config.ENTRY_VOTES,
            'predictions': {
                'momentum': {
                    'prediction': analysis['expert_opinions']['momentum']['prediction'],
                    'confidence': analysis['expert_opinions']['momentum']['confidence'],
                    'threshold': self.config.MIN_CONF_MOMENTUM,
                    'vote_eligible': analysis['expert_opinions']['momentum']['confidence'] >= self.config.MIN_CONF_MOMENTUM
                },
                'reversion': {
                    'prediction': analysis['expert_opinions']['reversion']['prediction'],
                    'confidence': analysis['expert_opinions']['reversion']['confidence'],
                    'threshold': self.config.MIN_CONF_REVERSION,
                    'vote_eligible': analysis['expert_opinions']['reversion']['confidence'] >= self.config.MIN_CONF_REVERSION
                },
                'pa': {
                    'prediction': analysis['expert_opinions']['pa']['prediction'],
                    'confidence': analysis['expert_opinions']['pa']['confidence'],
                    'threshold': self.config.MIN_CONF_PA,
                    'vote_eligible': analysis['expert_opinions']['pa']['confidence'] >= self.config.MIN_CONF_PA
                }
            },
            'recommendation': {
                'signal': 'LONG' if votes_long >= self.config.ENTRY_VOTES else 'SHORT' if votes_short >= self.config.ENTRY_VOTES else 'HOLD',
                'strength': max(votes_long, votes_short),
                'consensus': votes_long + votes_short >= 2,
                'conflicting': votes_long > 0 and votes_short > 0
            }
        }
    
    def get_position_status(self) -> Dict[str, Any]:
        """
        Returns current position status and management state.
        """
        if not self.active_position:
            return {
                'has_position': False,
                'position': None
            }
        
        pos = self.active_position
        return {
            'has_position': True,
            'position': {
                'strategy': pos.strategy,
                'entry_date': pos.entry_date,
                'entry_price': pos.entry_price,
                'size': pos.size,
                'current_sl_price': pos.current_sl_price,
                'tp_price': pos.tp_price,
                'is_be': pos.is_be,
                'is_trailing': pos.is_trailing,
                'breakeven_trigger_price': pos.breakeven_trigger_price,
                'trailing_trigger_price': pos.trailing_trigger_price,
                'opposing_signal_count': getattr(pos, 'opposing_signal_count', 0),
                'confidence_levels': {
                    'momentum': pos.conf_momentum,
                    'reversion': pos.conf_reversion,
                    'pa': pos.conf_pa
                }
            }
        }


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
        
        logger.info("🎯 APPLYING BREAK-EVEN")
        old_sl = pos.current_sl_price
        
        # Ustaw SL na poziom break-even
        if pos.strategy == 'long':
            if pos.breakeven_sl_price > pos.current_sl_price:
                pos.current_sl_price = pos.breakeven_sl_price
        else:  # short
            if pos.breakeven_sl_price < pos.current_sl_price:
                pos.current_sl_price = pos.breakeven_sl_price
        
        pos.is_be = True
        pos._pending_be = False
        logger.info(f"✅ Break-even applied: SL moved from {old_sl:.4f} to {pos.current_sl_price:.4f}")

    def _apply_pending_trailing_stop(self):
        """Stosuje oczekujący trailing stop."""
        pos = self.active_position
        if not hasattr(pos, "_pending_tsl_sl"):
            return
        
        candidate_sl = pos._pending_tsl_sl
        if candidate_sl is not None:
            old_sl = pos.current_sl_price
            updated = False
            
            if pos.strategy == 'long':
                if candidate_sl > pos.current_sl_price:
                    pos.current_sl_price = candidate_sl
                    updated = True
            else:  # short
                if candidate_sl < pos.current_sl_price:
                    pos.current_sl_price = candidate_sl
                    updated = True
            
            if updated:
                logger.info(f"📈 TRAILING STOP UPDATED: SL moved from {old_sl:.4f} to {pos.current_sl_price:.4f}")
            
            pos._pending_tsl_sl = None

    def _manage_active_position(self, current_candle, analysis) -> Optional[Dict[str, Any]]:
        """Zarządza aktywną pozycją i zwraca dane transakcji wyjścia jeśli pozycja zostanie zamknięta."""
        pos = self.active_position
        logger.info(f"=== MANAGING ACTIVE POSITION: {pos.strategy.upper()} ===")
        logger.info(f"Position entry: {pos.entry_price:.4f}, current SL: {pos.current_sl_price:.4f}, TP: {pos.tp_price:.4f}")
        logger.info(f"Position status - BE: {pos.is_be}, Trailing: {pos.is_trailing}")
        
        # Aktualizacja mechanik BE i TSL
        self._update_mechanics(current_candle)
        
        # Sprawdź sygnał wyjścia z modelu
        exit_reason = self._check_model_exit_signal(pos, analysis)
        
        # Sprawdź wyjście przez SL/TP (priorytet: SL przed TP)
        if not exit_reason:
            exit_reason = self._check_stop_take_exit(pos, current_candle)
        
        if exit_reason:
            logger.info(f"🚪 POSITION EXIT TRIGGERED: {exit_reason}")
            return self._create_exit_trade(pos, exit_reason, current_candle)
        else:
            logger.info("✅ Position continues - no exit conditions met")
        
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
        votes_long = self._count_votes(analysis, prediction_target=1)
        votes_short = self._count_votes(analysis, prediction_target=0)
        
        logger.info(f"🔍 CHECKING MODEL EXIT: Position {pos.strategy.upper()}, Long votes: {votes_long}, Short votes: {votes_short}")
        
        if pos.strategy == 'long':
            if votes_short >= 2:
                pos.opposing_signal_count += 1
                logger.info(f"⚠️ Opposing signal detected for LONG position! Count: {pos.opposing_signal_count}/{self.config.EXIT_SIGNAL_PERSISTENCE}")
            else:
                pos.opposing_signal_count = 0
                logger.info("✅ No opposing signals for LONG position")
                
            if pos.opposing_signal_count >= self.config.EXIT_SIGNAL_PERSISTENCE:
                logger.info("🚨 MODEL EXIT SIGNAL TRIGGERED for LONG position!")
                return "Model Exit Signal"
        elif pos.strategy == 'short':
            if votes_long >= 2:
                pos.opposing_signal_count += 1
                logger.info(f"⚠️ Opposing signal detected for SHORT position! Count: {pos.opposing_signal_count}/{self.config.EXIT_SIGNAL_PERSISTENCE}")
            else:
                pos.opposing_signal_count = 0
                logger.info("✅ No opposing signals for SHORT position")
                
            if pos.opposing_signal_count >= self.config.EXIT_SIGNAL_PERSISTENCE:
                logger.info("🚨 MODEL EXIT SIGNAL TRIGGERED for SHORT position!")
                return "Model Exit Signal"
        return None

    def _check_stop_take_exit(self, pos: Position, current_candle) -> Optional[str]:
        """Sprawdza czy pozycja powinna zostać zamknięta przez SL/TP."""
        logger.info(f"🎯 CHECKING SL/TP EXIT: {pos.strategy.upper()} position")
        logger.info(f"Current candle: High={current_candle['high']:.4f}, Low={current_candle['low']:.4f}")
        logger.info(f"Position levels: SL={pos.current_sl_price:.4f}, TP={pos.tp_price:.4f}")
        
        if pos.strategy == 'long':
            if current_candle['low'] <= pos.current_sl_price:
                exit_reason = self._classify_stop_reason(pos)
                logger.info(f"🔴 STOP LOSS HIT for LONG: {exit_reason}")
                return exit_reason
            elif current_candle['high'] >= pos.tp_price:
                logger.info("🟢 TAKE PROFIT HIT for LONG!")
                return "Take Profit"
        else:  # short
            if current_candle['high'] >= pos.current_sl_price:
                exit_reason = self._classify_stop_reason(pos)
                logger.info(f"🔴 STOP LOSS HIT for SHORT: {exit_reason}")
                return exit_reason
            elif current_candle['low'] <= pos.tp_price:
                logger.info("🟢 TAKE PROFIT HIT for SHORT!")
                return "Take Profit"
        
        logger.info("✅ No SL/TP exit conditions met")
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
        logger.info("=== CHECKING FOR NEW ENTRY ===")
        
        if not analysis:
            logger.info("❌ No analysis data available - no entry")
            return None
            
        if self.active_position:
            logger.info("❌ Active position exists - no new entry")
            return None

        strategy_to_open = self._determine_entry_strategy(analysis)
        if not strategy_to_open:
            logger.info("❌ No entry strategy determined - no entry")
            return None

        logger.info(f"✅ Entry strategy determined: {strategy_to_open.upper()}")
        position_params = self._calculate_position_parameters(current_candle, analysis, capital, strategy_to_open)
        logger.info(f"✅ Position parameters calculated - Size: {position_params['size']:.4f}, Entry: {position_params['entry_price']:.4f}, SL: {position_params['current_sl_price']:.4f}, TP: {position_params['tp_price']:.4f}")
        
        return position_params

    def _determine_entry_strategy(self, analysis) -> Optional[str]:
        """Określa strategię wejścia na podstawie głosów ekspertów."""
        votes_long = self._count_votes(analysis, prediction_target=1)
        votes_short = self._count_votes(analysis, prediction_target=0)
        
        logger.info(f"📊 ENTRY VOTING: Long votes: {votes_long}, Short votes: {votes_short}, Required: {self.config.ENTRY_VOTES}")
        
        if votes_long >= self.config.ENTRY_VOTES:
            logger.info("📈 LONG strategy selected based on expert voting")
            return 'long'
        elif votes_short >= self.config.ENTRY_VOTES:
            logger.info("📉 SHORT strategy selected based on expert voting")
            return 'short'
        else:
            logger.info("⚖️ No strategy selected - insufficient votes")
            return None

    def _calculate_position_parameters(self, current_candle, analysis, capital, strategy: str) -> Dict[str, Any]:
        """Oblicza wszystkie parametry nowej pozycji."""
        entry_price = analysis['current_price']
        sl_price, tp_price = self._calculate_sl_tp_prices(entry_price, analysis, strategy)
        position_size = self._calculate_position_size(entry_price, sl_price, capital)
        position_size = max(0.001, min(position_size, 0.1))
        be_trigger_price, be_sl_price = self._calculate_breakeven_params(entry_price, tp_price, strategy)
        tsl_trigger_price = self._calculate_trailing_params(entry_price, analysis, strategy)
        
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

    def _calculate_breakeven_params(self, entry_price: float, tp_price: float, strategy: str = 'long') -> Tuple[float, float]:
        """Oblicza parametry break-even."""
        tp_distance = abs(tp_price - entry_price)
        be_trigger = self.config.BREAKEVEN_TRIGGER_PERCENT
        
        if strategy == 'long':
            be_trigger_price = entry_price + (tp_distance * be_trigger) if be_trigger > 0 else 0
        else:  # short
            be_trigger_price = entry_price - (tp_distance * be_trigger) if be_trigger > 0 else 0
        
        be_sl_price = entry_price  # BE = entry (realne 0)
        return be_trigger_price, be_sl_price

    def _calculate_trailing_params(self, entry_price: float, analysis, strategy: str = 'long') -> float:
        """Oblicza parametry trailing stop."""
        stop_loss_distance = analysis['atr_value_5m'] * self.config.ATR_MULTIPLIER
        tsl_trigger = self.config.TRAILING_SL_TRIGGER_R
        
        if tsl_trigger <= 0:
            return 0
        
        if strategy == 'long':
            return entry_price + (stop_loss_distance * tsl_trigger)
        else:  # short
            return entry_price - (stop_loss_distance * tsl_trigger)

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
        
        else:  # short
            # 1) Trailing – aktywacja po R-multiple (stan), bez natychmiastowego podnoszenia SL
            if self.config.TRAILING_SL_TRIGGER_R > 0 and not pos.is_trailing and current_candle['low'] <= pos.trailing_trigger_price:
                pos.is_trailing = True
                self._log_event(current_candle.name, 'trailing_sl_activated', {'trade_entry_date': pos.entry_date})

            # 2) BE – tylko TRIGGER na tej świecy; zastosujemy na początku następnej
            elif self.config.BREAKEVEN_TRIGGER_PERCENT > 0 and not pos.is_be and current_candle['low'] <= pos.breakeven_trigger_price:
                pos._pending_be = True
                self._log_event(current_candle.name, 'breakeven_activated', {'trade_entry_date': pos.entry_date})

            # 3) Trailing – zapisujemy kandydata SL do zastosowania na N+1
            if pos.is_trailing:
                atr_here = current_candle['ATRr_14_5m']
                new_sl_candidate = current_candle['close'] + (atr_here * self.config.TRAILING_SL_DISTANCE_ATR)
                best = getattr(pos, "_pending_tsl_sl", None)
                if (best is None) or (new_sl_candidate < best):
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
