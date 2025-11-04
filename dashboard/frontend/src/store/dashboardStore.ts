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
  ExecutionQuality,
  FundingCosts,
  SLTPEffectiveness,
  SLTPTrendPoint,
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

  // Advanced Analytics
  executionQuality: ExecutionQuality | null;
  fundingCosts: FundingCosts | null;
  sltpEffectiveness: SLTPEffectiveness | null;
  sltpTrend: SLTPTrendPoint[];

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
  setExecutionQuality: (data: ExecutionQuality) => void;
  setFundingCosts: (data: FundingCosts) => void;
  setSLTPEffectiveness: (data: SLTPEffectiveness) => void;
  setSLTPTrend: (data: SLTPTrendPoint[]) => void;
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
  executionQuality: null,
  fundingCosts: null,
  sltpEffectiveness: null,
  sltpTrend: [],
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
  setExecutionQuality: (executionQuality) => set({ executionQuality }),
  setFundingCosts: (fundingCosts) => set({ fundingCosts }),
  setSLTPEffectiveness: (sltpEffectiveness) => set({ sltpEffectiveness }),
  setSLTPTrend: (sltpTrend) => set({ sltpTrend }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
  setSelectedTicker: (selectedTicker) => set({ selectedTicker }),
}));
