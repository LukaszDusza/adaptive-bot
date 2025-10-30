/**
 * TypeScript type definitions for Dashboard
 */

export interface MetricsResponse {
  total_pnl: number;
  total_pnl_percent: number;
  win_rate: number;
  total_trades: number;
  active_trades: number;
  sharpe_ratio: number | null;
  max_drawdown: number;
  max_drawdown_percent: number;
  profit_factor: number | null;
  avg_win: number;
  avg_loss: number;
  avg_trade_duration_hours: number;
  total_fees_paid: number;
  last_updated: string;
}

export interface TickerMetrics {
  ticker: string;
  total_pnl: number;
  win_rate: number;
  total_trades: number;
  avg_pnl: number;
  sharpe_ratio: number | null;
  max_drawdown: number;
  best_trade: number;
  worst_trade: number;
}

export interface Trade {
  trade_id: string;
  ticker: string;
  side: 'Long' | 'Short';
  start_time: string;
  end_time: string | null;
  entry_price: number | null;
  exit_price: number | null;
  quantity: number | null;
  leverage: number | null;
  initial_sl: number | null;
  initial_tp: number | null;
  current_sl: number | null;
  current_tp: number | null;
  summary: TradeSummary | null;
  events: TradeEvent[];
  indicators: TradeIndicators | null;
  is_active: boolean;
  notes: string | null;
}

export interface TradeSummary {
  pnl: number;
  pnl_percent: number;
  exit_reason: string;
  duration_seconds: number;
  max_favorable_excursion: number | null;
  max_adverse_excursion: number | null;
  fees_paid: number;
}

export interface TradeEvent {
  timestamp: string;
  type: string;
  data: Record<string, any>;
}

export interface TradeIndicators {
  entry: Record<string, any> | null;
  exit: Record<string, any> | null;
}

export interface ContainerInfo {
  name: string;
  container_id: string;
  status: 'running' | 'stopped' | 'restarting' | 'paused' | 'exited' | 'dead';
  ticker: string;
  account: string;
  strategy: string;
  uptime_seconds: number | null;
  health_status: string | null;
  restart_count: number;
}

export interface EquityCurvePoint {
  timestamp: string;
  cumulative_pnl: number;
  trade_count: number;
  drawdown: number;
}

export interface DrawdownPoint {
  timestamp: string;
  drawdown: number;
  drawdown_percent: number;
}

export interface ExitReasonStats {
  exit_reason: string;
  count: number;
  percentage: number;
  total_pnl: number;
}

export interface WebSocketMessage {
  event: string;
  data: any;
  timestamp: string;
}

export interface PendingOrder {
  order_id: string;
  ticker: string;
  side: 'Long' | 'Short';
  order_type: string;
  price: number;
  quantity: number;
  filled_quantity: number;
  status: string;
  created_at: string;
  time_in_force: string;
  reduce_only: boolean;
  stop_loss: number | null;
  take_profit: number | null;
}

export interface PendingOrdersResponse {
  orders: PendingOrder[];
  total_count: number;
  by_ticker: Record<string, number>;
  last_updated: string;
}
