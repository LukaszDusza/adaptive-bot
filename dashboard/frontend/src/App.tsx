/**
 * Main Dashboard Application
 */
import { useEffect, useState, useRef, useCallback } from 'react';
import { DashboardLayout } from './components/layout/DashboardLayout';
import { useMetrics } from './hooks/useMetrics';
import { MetricsCards } from './components/metrics/MetricsCards';
import { QuickStats } from './components/metrics/QuickStats';
import { ExecutionQualityCard } from './components/metrics/ExecutionQualityCard';
import { FundingCostsCard } from './components/metrics/FundingCostsCard';
import { SLTPEffectivenessCard } from './components/metrics/SLTPEffectivenessCard';
import { EquityCurveChart } from './components/charts/EquityCurveChart';
import { DrawdownChart } from './components/charts/DrawdownChart';
import { ExitReasonChart } from './components/charts/ExitReasonChart';
import { SLTPTrendChart } from './components/charts/SLTPTrendChart';
import { RecentTradesTable } from './components/tables/RecentTradesTable';
import { EmergencyCloseModal } from './components/controls/EmergencyCloseModal';
import { PendingOrdersPanel } from './components/controls/PendingOrdersPanel';
import { AlertBanner } from './components/alerts/AlertBanner';
import { ToastContainer } from './components/ui/Toast';
import { useDashboardStore } from './store/dashboardStore';
import { useWebSocket } from './hooks/useWebSocket';
import { useToast } from './hooks/useToast';
import { Shield, X, TrendingUp as ArrowUpCircle } from 'lucide-react';
import { LineChart, Line, ResponsiveContainer, YAxis } from 'recharts';
import {
  getOverallMetrics,
  getTickerMetrics,
  getAllTrades,
  getActiveTrades,
  getAllContainers,
  getEquityCurve,
  getDrawdownCurve,
  getExitReasonStats,
  getBatchPrices,
  getExecutionQuality,
  getFundingCosts,
  getSLTPEffectiveness,
  getSLTPEffectivenessTrend,
  closePosition,
  setStopLossToBreakeven,
  type MarketPrice,
} from './api/client';

function App() {
  const {
    metrics,
    tickerMetrics,
    trades,
    activeTrades,
    containers: _containers, // Prefixed with _ to indicate intentionally unused
    equityCurve,
    drawdownCurve,
    exitReasonStats,
    executionQuality,
    fundingCosts,
    sltpEffectiveness,
    sltpTrend,
    loading,
    error,
    setMetrics,
    setTickerMetrics,
    setTrades,
    setActiveTrades,
    setContainers,
    setEquityCurve,
    setDrawdownCurve,
    setExitReasonStats,
    setExecutionQuality,
    setFundingCosts,
    setSLTPEffectiveness,
    setSLTPTrend,
    setLoading,
    setError,
  } = useDashboardStore();

  // Local state for emergency modal
  const [showEmergencyModal, setShowEmergencyModal] = useState(false);

  // Ref for active trades section
  const activeTradesRef = useRef<HTMLDivElement>(null);

  // Market prices for active positions
  const [marketPrices, setMarketPrices] = useState<Record<string, MarketPrice>>({});

  // PnL history for mini chart (last 50 points)
  const [pnlHistory, setPnlHistory] = useState<Record<string, Array<{time: string, pnl: number}>>>({});

  // Toast notifications
  const toast = useToast();

  // Memoized metrics calculations (performance optimization)
  const { todayPnL, weeklyPnL } = useMetrics(trades);

  // Scroll to active trades (memoized callback)
  const scrollToActiveTrades = useCallback(() => {
    if (activeTradesRef.current) {
      activeTradesRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, []);

  // Position control handlers
  const handleClosePosition = async (symbol: string) => {
    if (!confirm(`Are you sure you want to close position for ${symbol}?`)) {
      return;
    }

    try {
      const result = await closePosition(symbol);
      toast.success(result.data.message);
      // Refresh active trades
      const activeTradesRes = await getActiveTrades();
      setActiveTrades(activeTradesRes.data);
    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || error.message || 'Unknown error';
      toast.error(`Failed to close position:\n${errorMsg}`);
    }
  };

  const handleSetSLtoBreakeven = async (symbol: string, side: string, entryPrice: number) => {
    if (!confirm(`Set stop loss to breakeven+fee for ${symbol}?`)) {
      return;
    }

    try {
      const result = await setStopLossToBreakeven(symbol, side, entryPrice);
      toast.success(result.data.message);

      // Wait 500ms for Bybit to update position data, then refresh
      await new Promise(resolve => setTimeout(resolve, 500));

      const activeTradesRes = await getActiveTrades();
      setActiveTrades(activeTradesRes.data);
    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || error.message || 'Unknown error';
      toast.error(`Failed to set stop loss:\n${errorMsg}`);
    }
  };

  // Connect to WebSocket
  useWebSocket();

  // Helper function to format PnL with dynamic precision
  const formatPnL = (value: number): string => {
    // For small PnL values (< $0.01), show 3 decimal places to avoid confusion
    const decimals = Math.abs(value) < 0.01 && value !== 0 ? 3 : 2;
    return value.toFixed(decimals);
  };

  // Fetch initial data
  const fetchData = async () => {
    setLoading(true);
    setError(null);

    try {
      const [
        metricsRes,
        tickerMetricsRes,
        tradesRes,
        activeTradesRes,
        containersRes,
        equityCurveRes,
        drawdownCurveRes,
        exitReasonStatsRes,
        executionQualityRes,
        fundingCostsRes,
        sltpEffectivenessRes,
        sltpTrendRes,
      ] = await Promise.all([
        getOverallMetrics(),
        getTickerMetrics(),
        getAllTrades(50),
        getActiveTrades(),
        getAllContainers(),
        getEquityCurve(),
        getDrawdownCurve(),
        getExitReasonStats(),
        getExecutionQuality(),
        getFundingCosts(30),
        getSLTPEffectiveness(7),
        getSLTPEffectivenessTrend(30),
      ]);

      setMetrics(metricsRes.data);
      setTickerMetrics(tickerMetricsRes.data);
      setTrades(tradesRes.data);
      setActiveTrades(activeTradesRes.data);
      setContainers(containersRes.data);
      setEquityCurve(equityCurveRes.data);
      setDrawdownCurve(drawdownCurveRes.data);
      setExitReasonStats(exitReasonStatsRes.data);
      setExecutionQuality(executionQualityRes.data);
      setFundingCosts(fundingCostsRes.data);
      setSLTPEffectiveness(sltpEffectivenessRes.data);
      setSLTPTrend(sltpTrendRes.data);
    } catch (err: any) {
      console.error('Failed to fetch dashboard data:', err);
      setError(err.message || 'Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();

    // Refresh data every 30 seconds
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  // Fetch current prices for active positions (OPTIMIZED: batch request)
  useEffect(() => {
    const fetchPrices = async () => {
      if (activeTrades.length === 0) return;

      try {
        // Extract unique tickers
        const uniqueTickers = [...new Set(activeTrades.map(t => t.ticker))];

        // Batch request - 1 API call instead of N
        const res = await getBatchPrices(uniqueTickers);
        const priceMap: Record<string, MarketPrice> = res.data.prices;

        // Update PnL history using functional update to avoid dependency issues
        setPnlHistory((prevHistory) => {
          const newPnlHistory = { ...prevHistory };

          activeTrades.forEach((trade) => {
            const price = priceMap[trade.ticker];
            if (price && trade.entry_price && trade.quantity) {
              // Calculate PnL
              let pnl = 0;
              if (trade.side === 'Long') {
                pnl = (price.last_price - trade.entry_price) * trade.quantity;
              } else {
                pnl = (trade.entry_price - price.last_price) * trade.quantity;
              }

              // Add to history (keep last 50 points)
              const tradeKey = trade.trade_id;
              if (!newPnlHistory[tradeKey]) {
                newPnlHistory[tradeKey] = [];
              }

              newPnlHistory[tradeKey] = [
                ...newPnlHistory[tradeKey],
                {
                  time: new Date().toLocaleTimeString(),
                  pnl: pnl
                }
              ].slice(-50); // Keep only last 50 points
            }
          });

          return newPnlHistory;
        });

        setMarketPrices(priceMap);
      } catch (err: any) {
        console.error('Failed to fetch batch prices:', err);
      }
    };

    fetchPrices();

    // Refresh prices every 5 seconds
    const interval = setInterval(fetchPrices, 5000);
    return () => clearInterval(interval);
  }, [activeTrades]);

  if (loading && !metrics) {
    return (
      <div className="h-full flex items-center justify-center bg-dark-bg">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-profit mx-auto"></div>
          <p className="mt-4 text-dark-text-secondary">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full flex items-center justify-center bg-dark-bg">
        <div className="card max-w-md">
          <h2 className="text-xl font-bold text-loss mb-4">Error Loading Dashboard</h2>
          <p className="text-dark-text-secondary mb-4">{error}</p>
          <button onClick={fetchData} className="btn btn-primary">
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <DashboardLayout
      activeTradesCount={activeTrades.length}
      onEmergencyClose={() => setShowEmergencyModal(true)}
      onPauseToggle={(paused) => {
        console.log(`Trading ${paused ? 'paused' : 'resumed'}`);
      }}
    >
      {/* Toast Notifications */}
      <ToastContainer toasts={toast.toasts} onRemove={toast.removeToast} />

      <div className="space-y-8">
        {/* Risk Alerts Banner */}
        <AlertBanner />

        {/* Quick Stats Bar (OPTIMIZED - memoized calculations) */}
        {metrics && (
          <QuickStats
            activeTrades={metrics.active_trades}
            tradesPerDay={metrics.trades_per_day}
            todayPnL={todayPnL}
            weeklyPnL={weeklyPnL}
            onActiveClick={scrollToActiveTrades}
          />
        )}

        {/* Main Metrics */}
        {metrics && <MetricsCards metrics={metrics} />}

        {/* Advanced Analytics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <ExecutionQualityCard data={executionQuality} loading={loading} />
          <FundingCostsCard data={fundingCosts} loading={loading} />
          <SLTPEffectivenessCard data={sltpEffectiveness} loading={loading} />
        </div>

        {/* SL/TP Effectiveness Trend */}
        <SLTPTrendChart data={sltpTrend} />

        {/* Pending Orders Panel */}
        <PendingOrdersPanel autoRefresh={true} refreshInterval={10000} />

        {/* Active Trades Summary */}
        {activeTrades.length > 0 && (
          <div ref={activeTradesRef} className="card">
            <h2 className="text-xl font-bold mb-4">Active Positions ({activeTrades.length})</h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {activeTrades.map((trade) => {
                // Parse partial_tp_taken - check for "partial_tp_taken=True" or events data
                const partialTpMatch = trade.notes?.match(/partial_tp_taken=(True|False)/);
                const partialTpTaken = partialTpMatch ? partialTpMatch[1] === 'True' :
                  (trade.events?.[0]?.data?.partial_tp_taken === true);

                // Parse dynamic_tp_levels_taken from events data
                const dynamicTpLevels = trade.events?.[0]?.data?.dynamic_tp_levels_taken ?? 0;

                // Get current price for display
                const currentPrice = marketPrices[trade.ticker]?.last_price;

                // Use PnL from backend (already calculated from Bybit API)
                // This is the REAL unrealized PnL from the exchange
                const unrealizedPnL = trade.summary?.pnl ?? null;
                const pnlPercentage = trade.summary?.pnl_percent ?? null;

                // Calculate potential PnL for TP and SL
                const tpPrice = trade.initial_tp || trade.current_tp;
                const slPrice = trade.current_sl || trade.initial_sl;
                const entryPrice = trade.entry_price;
                const quantity = trade.quantity || 0;
                const isLong = trade.side.toUpperCase() === 'LONG';

                // Check if position is secured (SL in profit zone)
                let isSecured = false;
                if (slPrice && entryPrice) {
                  if (isLong) {
                    // For LONG: SL is secured if it's above entry (in profit)
                    isSecured = slPrice > entryPrice;
                  } else {
                    // For SHORT: SL is secured if it's below entry (in profit)
                    isSecured = slPrice < entryPrice;
                  }
                }

                let potentialTpPnL = null;
                let potentialTpPnLPercent = null;
                let potentialSlPnL = null;
                let potentialSlPnLPercent = null;

                let positionValue = null;
                let marginUsed = null;
                const leverage = trade.leverage || 1;

                if (entryPrice && quantity > 0) {
                  const entryValue = entryPrice * quantity;
                  positionValue = entryValue; // Total position value (with leverage)
                  marginUsed = entryValue / leverage; // Actual capital engaged

                  if (tpPrice) {
                    if (isLong) {
                      potentialTpPnL = (tpPrice - entryPrice) * quantity;
                    } else {
                      potentialTpPnL = (entryPrice - tpPrice) * quantity;
                    }
                    potentialTpPnLPercent = (potentialTpPnL / entryValue) * 100;
                  }

                  if (slPrice) {
                    if (isLong) {
                      potentialSlPnL = (slPrice - entryPrice) * quantity;
                    } else {
                      potentialSlPnL = (entryPrice - slPrice) * quantity;
                    }
                    potentialSlPnLPercent = (potentialSlPnL / entryValue) * 100;
                  }
                }

                return (
                  <div
                    key={trade.trade_id}
                    className="p-4 bg-dark-bg rounded border border-dark-border"
                  >
                    <div className="flex items-center justify-between mb-3">
                      <div>
                        <span className="font-medium text-lg">{trade.ticker}</span>
                        <span className={`ml-2 badge ${trade.side.toUpperCase() === 'LONG' ? 'badge-success' : 'badge-danger'}`}>
                          {trade.side}
                        </span>
                      </div>
                      <div className="text-xs text-dark-text-secondary">
                        {new Date(trade.start_time).toLocaleString()}
                      </div>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                      <div>
                        <div className="text-dark-text-secondary">Entry Price</div>
                        <div className="font-mono font-medium">${trade.entry_price?.toFixed(5)}</div>
                      </div>
                      <div>
                        <div className="text-dark-text-secondary">Current Price</div>
                        <div className="font-mono font-medium">
                          {currentPrice ? (
                            <>${currentPrice.toFixed(5)}</>
                          ) : (
                            <span className="text-dark-text-secondary">Loading...</span>
                          )}
                        </div>
                      </div>
                      <div>
                        <div className="text-dark-text-secondary">Unrealized PnL</div>
                        {unrealizedPnL !== null ? (
                          <div className={`font-mono font-bold ${unrealizedPnL >= 0 ? 'text-profit' : 'text-loss'}`}>
                            ${formatPnL(unrealizedPnL)} ({pnlPercentage?.toFixed(2)}%)
                          </div>
                        ) : (
                          <div className="text-dark-text-secondary">-</div>
                        )}
                      </div>
                      <div>
                        <div className="text-dark-text-secondary">Quantity</div>
                        <div className="font-mono font-medium">{trade.quantity?.toFixed(1)}</div>
                      </div>
                      <div>
                        <div className="text-dark-text-secondary">
                          Margin Used
                          {leverage > 1 && (
                            <span className="text-xs ml-1">({leverage}x)</span>
                          )}
                        </div>
                        {marginUsed !== null ? (
                          <div>
                            <div className="font-mono font-medium text-blue-400">${marginUsed.toFixed(2)}</div>
                            {positionValue !== null && leverage > 1 && (
                              <div className="text-xs text-dark-text-secondary mt-0.5">
                                Position: ${positionValue.toFixed(2)}
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="text-dark-text-secondary">-</div>
                        )}
                      </div>
                      <div>
                        <div className="text-dark-text-secondary">TP Target</div>
                        {tpPrice ? (
                          <div>
                            <div className="font-mono font-medium text-profit">${tpPrice.toFixed(5)}</div>
                            {potentialTpPnL !== null && (
                              <div className="text-xs text-profit mt-0.5">
                                +${formatPnL(potentialTpPnL)} ({potentialTpPnLPercent?.toFixed(2)}%)
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="text-dark-text-secondary">-</div>
                        )}
                      </div>
                      <div>
                        <div className="text-dark-text-secondary">Stop Loss</div>
                        {slPrice ? (
                          <div>
                            <div className="font-mono font-medium text-loss">${slPrice.toFixed(5)}</div>
                            {potentialSlPnL !== null && (
                              <div className="text-xs text-loss mt-0.5">
                                {potentialSlPnL >= 0 ? '+' : ''}${formatPnL(potentialSlPnL)} ({potentialSlPnLPercent?.toFixed(2)}%)
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="text-dark-text-secondary">-</div>
                        )}
                      </div>
                    </div>

                    {/* Limit Orders Status & PnL Chart */}
                    <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                      {/* Limit Orders */}
                      <div className="p-3 bg-dark-card rounded border border-dark-border">
                        <div className="text-xs font-semibold text-dark-text-secondary mb-2">LIMIT ORDERS STATUS</div>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div>
                            <div className="text-dark-text-secondary text-xs">Partial TP</div>
                            <div className="font-mono font-bold text-xs">
                              {partialTpTaken ? (
                                <span className="text-profit">✓ Taken</span>
                              ) : (
                                <span className="text-gray-500">Pending</span>
                              )}
                            </div>
                          </div>
                          <div>
                            <div className="text-dark-text-secondary text-xs">Dynamic TP</div>
                            <div className="font-mono font-bold text-xs">
                              {dynamicTpLevels > 0 ? (
                                <span className="text-profit">{dynamicTpLevels}/4 taken</span>
                              ) : (
                                <span className="text-gray-500">0/4 taken</span>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* PnL Mini Chart */}
                      <div className="p-3 bg-dark-card rounded border border-dark-border">
                        <div className="text-xs font-semibold text-dark-text-secondary mb-2">UNREALIZED PNL TREND</div>
                        {pnlHistory[trade.trade_id] && pnlHistory[trade.trade_id].length > 1 ? (
                          <ResponsiveContainer width="100%" height={60}>
                            <LineChart data={pnlHistory[trade.trade_id]}>
                              <YAxis hide domain={['dataMin', 'dataMax']} />
                              <Line
                                type="monotone"
                                dataKey="pnl"
                                stroke={unrealizedPnL && unrealizedPnL >= 0 ? '#10b981' : '#ef4444'}
                                strokeWidth={2}
                                dot={false}
                                animationDuration={300}
                              />
                            </LineChart>
                          </ResponsiveContainer>
                        ) : (
                          <div className="h-[60px] flex items-center justify-center text-xs text-dark-text-secondary">
                            Collecting data...
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Position Control Buttons */}
                    <div className="flex gap-2 mt-4">
                      <button
                        onClick={() => handleSetSLtoBreakeven(trade.ticker, trade.side, entryPrice || 0)}
                        disabled={isSecured || !entryPrice}
                        className={`px-3 py-1.5 rounded text-sm font-medium flex items-center gap-1 ${
                          isSecured || !entryPrice
                            ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                            : 'bg-blue-600 hover:bg-blue-700 text-white'
                        }`}
                        title={isSecured ? 'SL already at breakeven or in profit' : 'Set stop loss to breakeven'}
                      >
                        <ArrowUpCircle size={14} />
                        SL → BE
                      </button>
                      <button
                        onClick={() => handleClosePosition(trade.ticker)}
                        className="px-3 py-1.5 rounded text-sm font-medium bg-red-600 hover:bg-red-700 text-white flex items-center gap-1"
                        title="Close this position immediately"
                      >
                        <X size={14} />
                        Close Position
                      </button>
                    </div>

                    {/* Status badges */}
                    <div className="flex gap-2 mt-3 flex-wrap">
                      {isSecured && (
                        <span className="badge bg-blue-600 flex items-center gap-1">
                          <Shield size={12} />
                          Secured
                        </span>
                      )}
                      {partialTpTaken && (
                        <span className="badge badge-success">Partial TP ✓</span>
                      )}
                      {!partialTpTaken && (
                        <span className="badge bg-gray-600">No TP yet</span>
                      )}
                    </div>

                    {/* Real-time update indicator */}
                    <div className="mt-3 text-xs text-dark-text-secondary italic flex items-center gap-1">
                      <span className="inline-block w-2 h-2 bg-profit rounded-full animate-pulse"></span>
                      Live prices updating every 5 seconds
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Per-Ticker Metrics */}
        {tickerMetrics.length > 0 && (
          <div className="card">
            <h2 className="text-xl font-bold mb-4">Per-Ticker Performance</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {tickerMetrics.map((tm) => (
                <div key={tm.ticker} className="p-4 bg-dark-bg rounded border border-dark-border">
                  <div className="text-lg font-bold mb-2">{tm.ticker}</div>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-dark-text-secondary">PnL:</span>
                      <span className={tm.total_pnl >= 0 ? 'text-profit' : 'text-loss'}>
                        ${tm.total_pnl.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-dark-text-secondary">Win Rate:</span>
                      <span>{(tm.win_rate * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-dark-text-secondary">Trades:</span>
                      <span>{tm.total_trades}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Charts Grid - 2 per row */}
        {(equityCurve.length > 0 || drawdownCurve.length > 0) && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Equity Curve Chart */}
            {equityCurve.length > 0 && <EquityCurveChart data={equityCurve} />}

            {/* Drawdown Chart */}
            {drawdownCurve.length > 0 && <DrawdownChart data={drawdownCurve} />}
          </div>
        )}

        {/* Exit Reason Breakdown */}
        {exitReasonStats.length > 0 && <ExitReasonChart data={exitReasonStats} />}

        {/* Recent Trades Table */}
        {trades.length > 0 && <RecentTradesTable trades={trades} />}

        {/* Footer Info */}
        <div className="text-center text-sm text-dark-text-secondary py-8">
          <p>Adaptive Bot Dashboard • FastAPI + React + TypeScript</p>
          <p className="mt-1">
            Last updated: {metrics?.last_updated ? new Date(metrics.last_updated).toLocaleString() : 'N/A'}
          </p>
        </div>
      </div>

      {/* Emergency Close Modal */}
      <EmergencyCloseModal
        isOpen={showEmergencyModal}
        onClose={() => setShowEmergencyModal(false)}
        onSuccess={fetchData}
        activePositionCount={activeTrades.length}
      />
    </DashboardLayout>
  );
}

export default App;
