/**
 * Zustand store for dashboard state
 */
import { create } from 'zustand';
import type {
  MetricsResponse,
  TickerMetrics,
  Trade,
  ContainerInfo,
  EquityCurvePoint,
  DrawdownPoint,
  ExitReasonStats,
} from '../types';

interface DashboardState {
  // Data
  metrics: MetricsResponse | null;
  tickerMetrics: TickerMetrics[];
  trades: Trade[];
  activeTrades: Trade[];
  containers: ContainerInfo[];
  equityCurve: EquityCurvePoint[];
  drawdownCurve: DrawdownPoint[];
  exitReasonStats: ExitReasonStats[];

  // UI State
  loading: boolean;
  error: string | null;
  selectedTicker: string | null;

  // Actions
  setMetrics: (metrics: MetricsResponse) => void;
  setTickerMetrics: (tickerMetrics: TickerMetrics[]) => void;
  setTrades: (trades: Trade[]) => void;
  setActiveTrades: (trades: Trade[]) => void;
  setContainers: (containers: ContainerInfo[]) => void;
  setEquityCurve: (curve: EquityCurvePoint[]) => void;
  setDrawdownCurve: (curve: DrawdownPoint[]) => void;
  setExitReasonStats: (stats: ExitReasonStats[]) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setSelectedTicker: (ticker: string | null) => void;
}

export const useDashboardStore = create<DashboardState>((set) => ({
  // Initial state
  metrics: null,
  tickerMetrics: [],
  trades: [],
  activeTrades: [],
  containers: [],
  equityCurve: [],
  drawdownCurve: [],
  exitReasonStats: [],
  loading: false,
  error: null,
  selectedTicker: null,

  // Actions
  setMetrics: (metrics) => set({ metrics }),
  setTickerMetrics: (tickerMetrics) => set({ tickerMetrics }),
  setTrades: (trades) => set({ trades }),
  setActiveTrades: (activeTrades) => set({ activeTrades }),
  setContainers: (containers) => set({ containers }),
  setEquityCurve: (equityCurve) => set({ equityCurve }),
  setDrawdownCurve: (drawdownCurve) => set({ drawdownCurve }),
  setExitReasonStats: (exitReasonStats) => set({ exitReasonStats }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
  setSelectedTicker: (selectedTicker) => set({ selectedTicker }),
}));
