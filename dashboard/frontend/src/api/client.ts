/**
 * API client for Dashboard backend
 */
import axios from 'axios';
import type {
  MetricsResponse,
  TickerMetrics,
  Trade,
  ContainerInfo,
  EquityCurvePoint,
  DrawdownPoint,
  ExitReasonStats,
  PendingOrdersResponse,
  ExecutionQuality,
  FundingCosts,
  SLTPEffectiveness,
} from '../types';

// Determine base URL: empty string for production (proxied by nginx), full URL for dev
const rawUrl = import.meta.env.VITE_API_URL;
const API_BASE_URL = rawUrl === '' || rawUrl === undefined || rawUrl === 'undefined'
  ? '' // Production: nginx proxy handles /api routing
  : rawUrl; // Development: direct connection to backend

// Log for debugging
console.log('VITE_API_URL:', import.meta.env.VITE_API_URL, 'Resolved API_BASE_URL:', API_BASE_URL);

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Health
export const getHealth = () => api.get('/api/health');

// Metrics
export const getOverallMetrics = () =>
  api.get<MetricsResponse>('/api/metrics/overall');

export const getTickerMetrics = () =>
  api.get<TickerMetrics[]>('/api/metrics/tickers');

export const getEquityCurve = () =>
  api.get<EquityCurvePoint[]>('/api/metrics/equity-curve');

export const getDrawdownCurve = () =>
  api.get<DrawdownPoint[]>('/api/metrics/drawdown-curve');

export const getExitReasonStats = () =>
  api.get<ExitReasonStats[]>('/api/metrics/exit-reasons');

// Alerts
export interface Alert {
  severity: 'info' | 'warning' | 'critical';
  title: string;
  message: string;
  action?: string;
  timestamp: string;
  metric_value?: number;
  threshold?: number;
}

export const getRiskAlerts = () =>
  api.get<Alert[]>('/api/metrics/alerts');

// Trades
export const getAllTrades = (limit?: number, ticker?: string) =>
  api.get<Trade[]>('/api/trades/', { params: { limit, ticker } });

export const getActiveTrades = () =>
  api.get<Trade[]>('/api/trades/active');

export const getTradeById = (tradeId: string) =>
  api.get<Trade>(`/api/trades/${tradeId}`);

export const updateTradeNote = (tradeId: string, note: string) =>
  api.patch(`/api/trades/${tradeId}/note`, { note });

export const getPendingOrders = (tickers?: string) =>
  api.get<PendingOrdersResponse>('/api/trades/pending-orders', { params: { tickers } });

export const cancelPendingOrder = (ticker: string, orderId: string) =>
  api.delete(`/api/trades/pending-orders/${ticker}/${orderId}`);

export const cancelAllPendingOrders = (ticker: string) =>
  api.delete(`/api/trades/pending-orders/${ticker}`);

// Containers
export const getAllContainers = () =>
  api.get<ContainerInfo[]>('/api/containers/');

export const startContainer = (name: string) =>
  api.post(`/api/containers/${name}/start`);

export const stopContainer = (name: string) =>
  api.post(`/api/containers/${name}/stop`);

export const restartContainer = (name: string) =>
  api.post(`/api/containers/${name}/restart`);

export const getContainerLogs = (name: string, tail = 100) =>
  api.get(`/api/containers/${name}/logs`, { params: { tail } });

// Bot Control
export const pauseTrading = (paused: boolean, tickers?: string[], reason?: string) =>
  api.post('/api/bot-control/pause', { paused, tickers, reason });

export const emergencyCloseAll = (confirmation: boolean, tickers?: string[], reason = 'Emergency close via dashboard') =>
  api.post('/api/bot-control/emergency-close', { confirmation, tickers, reason });

export const getPauseState = () =>
  api.get<Record<string, boolean>>('/api/bot-control/pause-state');

// Market Data
export interface MarketPrice {
  symbol: string;
  last_price: number;
  mark_price: number;
  index_price: number;
  price_24h_change: string;
}

export const getCurrentPrice = (symbol: string) =>
  api.get<MarketPrice>(`/api/market/price/${symbol}`);

export const getMultiplePrices = (symbols: string) =>
  api.get<Record<string, MarketPrice>>(`/api/market/prices?symbols=${symbols}`);

/**
 * Get batch prices for multiple symbols (OPTIMIZED - parallel fetching)
 *
 * Reduces N API calls to 1 batch request.
 * Up to 10x faster than sequential getCurrentPrice calls.
 */
export interface BatchPriceResponse {
  prices: Record<string, MarketPrice>;
  errors: string[];
  total: number;
  successful: number;
  failed: number;
}

export const getBatchPrices = (symbols: string[]) =>
  api.post<BatchPriceResponse>('/api/market/batch', { symbols });

// Advanced Analytics
export const getExecutionQuality = () =>
  api.get<ExecutionQuality>('/api/metrics/execution-quality');

export const getFundingCosts = (days: number = 30) =>
  api.get<FundingCosts>('/api/metrics/funding-costs', { params: { days } });

export const getSLTPEffectiveness = () =>
  api.get<SLTPEffectiveness>('/api/metrics/sl-tp-effectiveness');

export default api;
