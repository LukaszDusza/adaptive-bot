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
import { DockerPanel } from './components/controls/DockerPanel';
import { EquityCurveChart } from './components/charts/EquityCurveChart';
import { DrawdownChart } from './components/charts/DrawdownChart';
import { ExitReasonChart } from './components/charts/ExitReasonChart';
import { RecentTradesTable } from './components/tables/RecentTradesTable';
import { EmergencyCloseModal } from './components/controls/EmergencyCloseModal';
import { PauseResumeToggle } from './components/controls/PauseResumeToggle';
import { PendingOrdersPanel } from './components/controls/PendingOrdersPanel';
import { AlertBanner } from './components/alerts/AlertBanner';
import { useDashboardStore } from './store/dashboardStore';
import { useWebSocket } from './hooks/useWebSocket';
import { AlertTriangle, Shield } from 'lucide-react';
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
  type MarketPrice,
} from './api/client';

function App() {
  const {
    metrics,
    tickerMetrics,
    trades,
    activeTrades,
    containers,
    equityCurve,
    drawdownCurve,
    exitReasonStats,
    executionQuality,
    fundingCosts,
    sltpEffectiveness,
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

  // Memoized metrics calculations (performance optimization)
  const { todayPnL, weeklyPnL } = useMetrics(trades);

  // Scroll to active trades (memoized callback)
  const scrollToActiveTrades = useCallback(() => {
    if (activeTradesRef.current) {
      activeTradesRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, []);

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
        getSLTPEffectiveness(),
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
    <DashboardLayout>
      <div className="space-y-8">
        {/* Risk Alerts Banner */}
        <AlertBanner />

        {/* Header with Controls */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">Dashboard Overview</h1>
            <p className="text-dark-text-secondary">
              Real-time monitoring of your trading bots
            </p>
          </div>

          {/* Control Buttons */}
          <div className="flex items-center gap-3">
            {/* Pause/Resume Toggle */}
            <PauseResumeToggle onStateChange={(paused) => {
              console.log(`Trading ${paused ? 'paused' : 'resumed'}`);
            }} />

            {/* Emergency Close Button */}
            {activeTrades.length > 0 && (
              <button
                onClick={() => setShowEmergencyModal(true)}
                className="btn btn-danger flex items-center gap-2 animate-pulse-slow"
              >
                <AlertTriangle size={16} />
                Emergency Close ({activeTrades.length})
              </button>
            )}
          </div>
        </div>

        {/* Quick Stats Bar (OPTIMIZED - memoized calculations) */}
        {metrics && (
          <QuickStats
            totalTrades={metrics.total_trades}
            activeTrades={metrics.active_trades}
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

        {/* Pending Orders Panel */}
        <PendingOrdersPanel autoRefresh={true} refreshInterval={10000} />

        {/* Active Trades Summary */}
        {activeTrades.length > 0 && (
          <div ref={activeTradesRef} className="card">
            <h2 className="text-xl font-bold mb-4">Active Positions ({activeTrades.length})</h2>
            <div className="space-y-2">
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

                // Check if position is secured (SL at breakeven)
                const isSecured = trade.current_sl && trade.entry_price &&
                  Math.abs(trade.current_sl - trade.entry_price) / trade.entry_price < 0.001; // Within 0.1%

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

                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
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
                        <div className="text-dark-text-secondary">TP Target</div>
                        <div className="font-mono font-medium text-profit">${trade.initial_tp?.toFixed(5)}</div>
                      </div>
                      <div>
                        <div className="text-dark-text-secondary">Stop Loss</div>
                        <div className="font-mono font-medium text-loss">${trade.current_sl?.toFixed(5)}</div>
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

        {/* Equity Curve Chart */}
        {equityCurve.length > 0 && <EquityCurveChart data={equityCurve} />}

        {/* Drawdown Chart */}
        {drawdownCurve.length > 0 && <DrawdownChart data={drawdownCurve} />}

        {/* Exit Reason Breakdown */}
        {exitReasonStats.length > 0 && <ExitReasonChart data={exitReasonStats} />}

        {/* Recent Trades Table */}
        {trades.length > 0 && <RecentTradesTable trades={trades} />}

        {/* Docker Containers */}
        <DockerPanel containers={containers} onRefresh={fetchData} />

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
