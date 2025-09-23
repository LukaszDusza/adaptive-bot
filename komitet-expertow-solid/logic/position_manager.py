# logic/position_manager.py
import logging
from .position import Position
from .fees import get_fee_calculator
from typing import Dict, Any, Optional, Tuple, List

# Setup logging
logger = logging.getLogger(__name__)


class PositionManager:
    def __init__(self, config):
        self.config = config
        self.active_position: Optional[Position] = None
        self.events = []
        self.fee_calculator = get_fee_calculator(config)
        self.expert_names = ['momentum', 'reversion', 'pa']

    def _log_event(self, timestamp, event_type, details):
        """Logs an event to the internal event list."""
        self.events.append({"timestamp": timestamp, "event": event_type, "details": details})

    # ==============================================================================
    # SECTION: UNIFIED TRADING LOGIC ENGINE
    # ==============================================================================

    def _evaluate_trading_actions(self, current_candle, analysis, capital) -> List[Dict[str, Any]]:
        """
        Core logic engine that evaluates the current state and returns a list of actions.
        This is used by both live trading and backtesting.

        Possible actions: 'CLOSE_POSITION', 'OPEN_POSITION', 'UPDATE_SL', 'ACTIVATE_TRAILING'.
        """
        actions = []

        # 1. First, check management actions for an active position
        if self.active_position:
            management_actions = self._check_position_management(current_candle, analysis)
            actions.extend(management_actions)

            # 2. Check for an exit signal (Model or SL/TP)
            exit_reason = self._check_model_exit_signal(self.active_position, analysis)
            if not exit_reason:
                exit_reason = self._check_stop_take_exit(self.active_position, current_candle)

            if exit_reason:
                # Clear other management actions if we are closing
                actions = [{
                    'type': 'CLOSE_POSITION',
                    'exit_reason': exit_reason,
                    'exit_price': self._get_raw_exit_price(exit_reason, self.active_position, current_candle)
                }]
                return actions

        # 3. If there's no active position (or it wasn't closed), check for a new entry
        if not self.active_position:
            strategy = self._determine_entry_strategy(analysis)
            if strategy:
                position_params = self._calculate_position_parameters(current_candle, analysis, capital, strategy)
                actions.append({
                    'type': 'OPEN_POSITION',
                    **position_params
                })

        return actions

    # ==============================================================================
    # SECTION: LIVE TRADING API METHODS
    # ==============================================================================

    def get_trading_signal(self, current_candle, analysis, capital) -> Dict[str, Any]:
        """
        API method for live trading. Evaluates the situation and returns a signal dictionary.
        """
        self._log_ml_predictions(analysis)

        actions = self._evaluate_trading_actions(current_candle, analysis, capital)

        # Translate actions into the required API format
        if not actions:
            return self._create_api_response('HOLD')

        # Prioritize CLOSE action
        close_action = next((a for a in actions if a['type'] == 'CLOSE_POSITION'), None)
        if close_action:
            return self._create_api_response(
                'CLOSE',
                exit_reason=close_action['exit_reason'],
                exit_price=close_action['exit_price']
            )

        # Prioritize OPEN action
        open_action = next((a for a in actions if a['type'] == 'OPEN_POSITION'), None)
        if open_action:
            return self._create_api_response(
                f'OPEN_{open_action["strategy"].upper()}',
                strategy=open_action['strategy'],
                entry_price=open_action['entry_price'],
                stop_loss=open_action['current_sl_price'],
                take_profit=open_action['tp_price'],
                size=open_action['size'],
                breakeven_trigger=open_action['breakeven_trigger_price'],
                trailing_trigger=open_action['trailing_trigger_price']
            )

        # If there are only management actions, return them as instructions
        instructions = self._extract_instructions_from_actions(actions)
        if instructions:
            return self._create_api_response('HOLD', instructions=instructions)

        return self._create_api_response('HOLD')

    def _create_api_response(self, action_name: str, **kwargs) -> Dict[str, Any]:
        """Helper to build the standard API response dictionary."""
        response = {'action': action_name, 'instructions': kwargs.get('instructions', [])}
        response.update(kwargs)
        return response

    def _extract_instructions_from_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Converts management actions into the 'instructions' format for the API."""
        instructions = []
        for action in actions:
            if action['type'] == 'MOVE_SL_TO_BREAKEVEN':
                instructions.append({
                    'type': 'MOVE_SL_TO_BREAKEVEN',
                    'new_sl_price': action['new_sl_price'],
                    'reason': 'Breakeven trigger reached'
                })
            elif action['type'] == 'ACTIVATE_TRAILING_STOP':
                instructions.append({
                    'type': 'ACTIVATE_TRAILING_STOP',
                    'reason': 'Trailing stop trigger reached'
                })
            elif action['type'] == 'UPDATE_TRAILING_STOP':
                instructions.append({
                    'type': 'UPDATE_TRAILING_STOP',
                    'new_sl_price': action['new_sl_price'],
                    'reason': 'Trailing stop update'
                })
        return instructions

    def confirm_breakeven(self):
        """Confirms that the break-even SL has been successfully set."""
        if self.active_position:
            self.active_position.is_be = True
            logger.info("✅ Break-even state confirmed internally.")

    def confirm_trailing_stop(self):
        """Confirms that the trailing stop has been successfully activated."""
        if self.active_position:
            self.active_position.is_trailing = True
            logger.info("✅ Trailing stop state confirmed internally.")

    def update_position_from_live_data(self, position_data: Dict[str, Any]):
        """
        Updates internal position state from live trading data,
        preserving internal management flags (is_be, is_trailing).
        """
        if not self.active_position:
            self.active_position = Position(**position_data)
            logger.info("Internal position created from live data.")
        else:
            is_be = self.active_position.is_be
            is_trailing = self.active_position.is_trailing

            for key, value in position_data.items():
                if hasattr(self.active_position, key):
                    setattr(self.active_position, key, value)

            self.active_position.is_be = is_be
            self.active_position.is_trailing = is_trailing
            logger.info("Internal position updated from live data, management flags preserved.")

    def clear_position(self):
        """Clears the active position (called when position is closed in live trading)."""
        self.active_position = None

    def get_ml_predictions(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns ML model predictions and confidence levels separately from trading decisions.
        """
        votes_long = self._count_votes(analysis, prediction_target=1)
        votes_short = self._count_votes(analysis, prediction_target=0)

        predictions = {}
        for name in self.expert_names:
            opinion = analysis['expert_opinions'][name]
            min_conf_key = f"MIN_CONF_{name.upper()}"
            min_conf_value = getattr(self.config, min_conf_key, 0.5)
            predictions[name] = {
                'prediction': opinion['prediction'],
                'confidence': opinion['confidence'],
                'threshold': min_conf_value,
                'vote_eligible': opinion['confidence'] >= min_conf_value
            }

        return {
            'votes_long': votes_long,
            'votes_short': votes_short,
            'total_experts': len(self.expert_names),
            'entry_threshold': self.config.ENTRY_VOTES,
            'predictions': predictions,
            'recommendation': {
                'signal': 'LONG' if votes_long >= self.config.ENTRY_VOTES else 'SHORT' if votes_short >= self.config.ENTRY_VOTES else 'HOLD',
                'strength': max(votes_long, votes_short),
                'consensus': votes_long + votes_short >= 2,
                'conflicting': votes_long > 0 and votes_short > 0
            }
        }

    def get_position_status(self) -> Dict[str, Any]:
        """Returns current position status and management state."""
        if not self.active_position:
            return {'has_position': False, 'position': None}

        pos = self.active_position
        confidence_levels = {name: getattr(pos, f"conf_{name}") for name in self.expert_names}

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
                'confidence_levels': confidence_levels
            }
        }

    # ==============================================================================
    # SECTION: BACKTESTING METHODS
    # ==============================================================================

    def process_candle(self, current_candle, analysis, capital) -> Tuple[Optional[str], Optional[Any]]:
        """
        Processes a single candle for backtesting.
        Returns: Tuple[action, details] where action is 'OPEN', 'CLOSE' or None.
        """
        actions = self._evaluate_trading_actions(current_candle, analysis, capital)

        for action in actions:
            if action['type'] == 'CLOSE_POSITION':
                if self.active_position:  # Ensure there's a position to close
                    closed_trade = self._create_exit_trade(self.active_position, action['exit_reason'], current_candle)
                    self.active_position = None
                    return 'CLOSE', closed_trade

            if action['type'] == 'OPEN_POSITION':
                position_params = action.copy()
                del position_params['type']

                self.active_position = Position(**position_params)

                management_actions = self._check_position_management(current_candle, analysis)
                for mgmt_action in management_actions:
                    self._apply_management_action_for_backtest(mgmt_action, current_candle)
                return 'OPEN', self.active_position

            if self.active_position:
                self._apply_management_action_for_backtest(action, current_candle)

        return None, None

    def _apply_management_action_for_backtest(self, action: Dict[str, Any], current_candle):
        """Applies a management action to the active position during a backtest."""
        pos = self.active_position
        if not pos: return

        action_type = action.get('type')
        if action_type == 'MOVE_SL_TO_BREAKEVEN':
            if not pos.is_be:
                pos.current_sl_price = action['new_sl_price']
                pos.is_be = True
                logger.info(f"✅ (BT) Break-even applied: SL moved to {pos.current_sl_price:.4f}")
                self._log_event(current_candle.name, 'breakeven_applied', {'trade_entry_date': pos.entry_date})

        elif action_type == 'ACTIVATE_TRAILING_STOP':
            if not pos.is_trailing:
                pos.is_trailing = True
                logger.info("📈 (BT) Trailing stop activated.")
                self._log_event(current_candle.name, 'trailing_sl_activated', {'trade_entry_date': pos.entry_date})

        elif action_type == 'UPDATE_TRAILING_STOP':
            old_sl = pos.current_sl_price
            if (pos.strategy == 'long' and action['new_sl_price'] > old_sl) or \
                    (pos.strategy == 'short' and action['new_sl_price'] < old_sl):
                pos.current_sl_price = action['new_sl_price']
                logger.info(f"📈 (BT) TRAILING STOP UPDATED: SL moved from {old_sl:.4f} to {pos.current_sl_price:.4f}")

    # ==============================================================================
    # SECTION: CORE LOGIC & HELPER METHODS
    # ==============================================================================

    def _log_ml_predictions(self, analysis: Dict[str, Any]):
        """Logs ML predictions and expert opinions."""
        logger.info("=== ML PREDICTIONS ===")
        logger.info(f"Current price: {analysis['current_price']:.4f}")
        logger.info(f"ATR (5m): {analysis['atr_value_5m']:.6f}")

        for expert_name, opinion in analysis['expert_opinions'].items():
            prediction_text = "BULLISH" if opinion['prediction'] == 1 else "BEARISH"
            logger.info(f"{expert_name.upper()}: {prediction_text} (confidence: {opinion['confidence']:.4f})")

    def _check_position_management(self, current_candle, analysis) -> List[Dict[str, Any]]:
        """
        Checks for and returns a list of position management actions (BE, TSL).
        """
        pos = self.active_position
        if not pos: return []

        actions = []

        if self.config.BREAKEVEN_TRIGGER_PERCENT > 0 and not pos.is_be and not pos.is_trailing:
            if (pos.strategy == 'long' and current_candle['high'] >= pos.breakeven_trigger_price) or \
                    (pos.strategy == 'short' and current_candle['low'] <= pos.breakeven_trigger_price):
                actions.append({
                    'type': 'MOVE_SL_TO_BREAKEVEN',
                    'new_sl_price': pos.breakeven_sl_price
                })

        elif self.config.TRAILING_SL_TRIGGER_R > 0 and not pos.is_trailing:
            if (pos.strategy == 'long' and current_candle['high'] >= pos.trailing_trigger_price) or \
                    (pos.strategy == 'short' and current_candle['low'] <= pos.trailing_trigger_price):
                actions.append({'type': 'ACTIVATE_TRAILING_STOP'})

        if pos.is_trailing or any(a['type'] == 'ACTIVATE_TRAILING_STOP' for a in actions):
            atr_here = analysis.get('atr_value_5m', current_candle.get('ATRr_14_5m'))

            if pos.strategy == 'long':
                new_sl_candidate = current_candle['close'] - (atr_here * self.config.TRAILING_SL_DISTANCE_ATR)
                if new_sl_candidate > pos.current_sl_price:
                    actions.append({
                        'type': 'UPDATE_TRAILING_STOP',
                        'new_sl_price': new_sl_candidate
                    })
            else:
                new_sl_candidate = current_candle['close'] + (atr_here * self.config.TRAILING_SL_DISTANCE_ATR)
                if new_sl_candidate < pos.current_sl_price:
                    actions.append({
                        'type': 'UPDATE_TRAILING_STOP',
                        'new_sl_price': new_sl_candidate
                    })

        return actions

    def _create_exit_trade(self, pos: Position, exit_reason: str, current_candle) -> Dict[str, Any]:
        """Creates the closed trade dictionary for logging/analysis."""
        raw_exit_price = self._get_raw_exit_price(exit_reason, pos, current_candle)
        exit_price = self.fee_calculator.apply_slippage(
            exit_reason, pos.strategy, raw_exit_price, current_candle
        )
        pnl = self.fee_calculator.calculate_pnl(
            pos.strategy, pos.entry_price, exit_price, pos.size
        )

        confidence_levels = {name: getattr(pos, f"conf_{name}") for name in self.expert_names}

        return {
            'entry_date': pos.entry_date,
            'exit_date': current_candle.name,
            'entry_price': pos.entry_price,
            'exit_price': exit_price,
            'size': pos.size,
            'pnl_usd': pnl,
            'exit_reason': exit_reason,
            'strategy': pos.strategy,
            **{f"conf_{name}": conf for name, conf in confidence_levels.items()}
        }

    def _get_raw_exit_price(self, exit_reason: str, pos: Position, current_candle) -> float:
        """Returns the raw exit price before applying slippage."""
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
        """Checks if a model-based exit signal has occurred."""
        votes_long = self._count_votes(analysis, prediction_target=1)
        votes_short = self._count_votes(analysis, prediction_target=0)

        exit_votes_threshold = getattr(self.config, 'EXIT_VOTES', 2)
        logger.info(
            f"🔍 CHECKING MODEL EXIT: Position {pos.strategy.upper()}, Long votes: {votes_long}, Short votes: {votes_short}, Exit Threshold: {exit_votes_threshold}")

        is_opposing_signal = (pos.strategy == 'long' and votes_short >= exit_votes_threshold) or \
                             (pos.strategy == 'short' and votes_long >= exit_votes_threshold)

        if is_opposing_signal:
            pos.opposing_signal_count += 1
            logger.warning(
                f"⚠️ Opposing signal detected! Count: {pos.opposing_signal_count}/{self.config.EXIT_SIGNAL_PERSISTENCE}")
        else:
            if pos.opposing_signal_count > 0:
                logger.info("✅ Opposing signal count reset to 0.")
            pos.opposing_signal_count = 0

        if pos.opposing_signal_count >= self.config.EXIT_SIGNAL_PERSISTENCE:
            logger.info(f"🚨 MODEL EXIT SIGNAL TRIGGERED for {pos.strategy.upper()} position!")
            return "Model Exit Signal"

        return None

    def _check_stop_take_exit(self, pos: Position, current_candle) -> Optional[str]:
        """Checks if the position should be closed due to Stop Loss or Take Profit."""
        logger.info(
            f"🎯 CHECKING SL/TP EXIT: High={current_candle['high']:.4f}, Low={current_candle['low']:.4f}, SL={pos.current_sl_price:.4f}, TP={pos.tp_price:.4f}")

        if pos.strategy == 'long':
            if current_candle['low'] <= pos.current_sl_price:
                return self._classify_stop_reason(pos)
            if current_candle['high'] >= pos.tp_price:
                return "Take Profit"
        else:
            if current_candle['high'] >= pos.current_sl_price:
                return self._classify_stop_reason(pos)
            if current_candle['low'] <= pos.tp_price:
                return "Take Profit"

        return None

    def _classify_stop_reason(self, pos: Position) -> str:
        """Classifies the reason for a stop loss exit."""
        if pos.is_be:
            return "Break-Even"
        if pos.is_trailing:
            return "Trailing Stop"
        return "Stop Loss"

    def _determine_entry_strategy(self, analysis) -> Optional[str]:
        """Determines entry strategy based on expert votes."""
        if not analysis:
            logger.warning("No analysis data available - no entry.")
            return None

        votes_long = self._count_votes(analysis, prediction_target=1)
        votes_short = self._count_votes(analysis, prediction_target=0)

        logger.info(
            f"📊 ENTRY VOTING: Long votes: {votes_long}, Short votes: {votes_short}, Required: {self.config.ENTRY_VOTES}")

        if (votes_long >= self.config.ENTRY_VOTES): return 'long'
                # and votes_short == 0):

        # if votes_short >= self.config.ENTRY_VOTES and votes_long == 0:
        #     return 'short'

        return None

    def _calculate_position_parameters(self, current_candle, analysis, capital, strategy: str) -> Dict[str, Any]:
        """Calculates all parameters for a new position."""
        entry_price = analysis['current_price']
        sl_price, tp_price = self._calculate_sl_tp_prices(entry_price, analysis, strategy)

        risk_per_unit = abs(entry_price - sl_price)
        risk_usd = capital * self.config.RISK_PERCENT
        position_size = (risk_usd / risk_per_unit) if risk_per_unit > 0 else 0

        min_size = getattr(self.config, "MIN_POSITION_SIZE", 0.001)
        max_size = getattr(self.config, "MAX_POSITION_SIZE", 0.1)
        position_size = max(min_size, min(position_size, max_size))

        be_trigger_price, be_sl_price = self._calculate_breakeven_params(entry_price, sl_price, strategy)
        tsl_trigger_price = self._calculate_trailing_params(entry_price, sl_price, strategy)

        confidence_params = {f"conf_{name}": analysis['expert_opinions'][name]['confidence'] for name in
                             self.expert_names}

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
            **confidence_params
        }

    def _calculate_sl_tp_prices(self, entry_price: float, analysis, strategy: str) -> Tuple[float, float]:
        """Calculates stop loss and take profit prices."""
        stop_loss_distance = analysis['atr_value_5m'] * self.config.ATR_MULTIPLIER

        if strategy == 'long':
            sl_price = entry_price - stop_loss_distance
            tp_price = entry_price + (stop_loss_distance * self.config.RRR)
        else:
            sl_price = entry_price + stop_loss_distance
            tp_price = entry_price - (stop_loss_distance * self.config.RRR)

        return sl_price, tp_price

    def _calculate_breakeven_params(self, entry_price: float, sl_price: float, strategy: str) -> Tuple[float, float]:
        """Calculates break-even parameters."""
        be_trigger_percent = getattr(self.config, "BREAKEVEN_TRIGGER_PERCENT", 0)
        if be_trigger_percent <= 0:
            return 0, 0

        sl_distance = abs(entry_price - sl_price)
        tp_distance = sl_distance * self.config.RRR

        if strategy == 'long':
            be_trigger_price = entry_price + (tp_distance * be_trigger_percent)
        else:
            be_trigger_price = entry_price - (tp_distance * be_trigger_percent)

        return be_trigger_price, entry_price

    def _calculate_trailing_params(self, entry_price: float, sl_price: float, strategy: str) -> float:
        """Calculates trailing stop trigger price."""
        tsl_trigger_r = getattr(self.config, "TRAILING_SL_TRIGGER_R", 0)
        if tsl_trigger_r <= 0:
            return 0

        stop_loss_distance = abs(entry_price - sl_price)

        if strategy == 'long':
            return entry_price + (stop_loss_distance * tsl_trigger_r)
        else:
            return entry_price - (stop_loss_distance * tsl_trigger_r)

    def _count_votes(self, analysis, prediction_target):
        """Counts the number of experts voting for a specific target."""
        votes = 0
        for expert_name in self.expert_names:
            min_conf_key = f"MIN_CONF_{expert_name.upper()}"
            min_conf_value = getattr(self.config, min_conf_key, 0.5)

            opinion = analysis['expert_opinions'].get(expert_name)
            if opinion and opinion['confidence'] >= min_conf_value and opinion['prediction'] == prediction_target:
                votes += 1
        return votes