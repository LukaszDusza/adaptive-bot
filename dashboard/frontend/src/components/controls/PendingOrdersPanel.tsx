/**
 * Pending Orders Panel - Shows limit orders waiting for execution
 * Features:
 * 1. Distance to Current Price with alerts
 * 2. Cancel Order buttons with confirmation
 * 3. DCA Level Indicators
 * 4. Risk Exposure Summary
 */
import { useEffect, useState } from 'react';
import {
  Clock, TrendingUp, TrendingDown, AlertCircle, RefreshCw, X,
  Trash2, DollarSign, Target, AlertTriangle, Shield
} from 'lucide-react';
import {
  getPendingOrders,
  cancelPendingOrder,
  cancelAllPendingOrders,
  getCurrentPrice
} from '../../api/client';
import type { PendingOrder, PendingOrdersResponse } from '../../types';

interface PendingOrdersPanelProps {
  tickers?: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

interface CancelConfirmation {
  type: 'single' | 'all';
  ticker: string;
  orderId?: string;
  orderPrice?: number;
}

interface DCAInfo {
  level: number; // 1, 2, 3
  type: string; // "Fixed offset", "ATR-based", "Swing distance"
}

export function PendingOrdersPanel({
  tickers,
  autoRefresh = true,
  refreshInterval = 10000
}: PendingOrdersPanelProps) {
  const [data, setData] = useState<PendingOrdersResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [cancelConfirm, setCancelConfirm] = useState<CancelConfirmation | null>(null);
  const [cancelling, setCancelling] = useState(false);

  // Current prices for distance calculation
  const [currentPrices, setCurrentPrices] = useState<Record<string, number>>({});

  const fetchOrders = async () => {
    setLoading(true);
    setError(null);

    try {
      const res = await getPendingOrders(tickers);
      setData(res.data);
      setLastUpdated(new Date());
    } catch (err: any) {
      console.error('Failed to fetch pending orders:', err);
      setError(err.message || 'Failed to load pending orders');
    } finally {
      setLoading(false);
    }
  };

  const fetchPrices = async () => {
    if (!data || data.orders.length === 0) return;

    const uniqueTickers = Array.from(new Set(data.orders.map(o => o.ticker)));
    const prices: Record<string, number> = {};

    for (const ticker of uniqueTickers) {
      try {
        const res = await getCurrentPrice(ticker);
        prices[ticker] = res.data.last_price;
      } catch (err) {
        console.error(`Failed to fetch price for ${ticker}:`, err);
      }
    }

    setCurrentPrices(prices);
  };

  useEffect(() => {
    fetchOrders();

    if (autoRefresh) {
      const interval = setInterval(fetchOrders, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [tickers, autoRefresh, refreshInterval]);

  // Fetch prices when orders change
  useEffect(() => {
    fetchPrices();

    // Refresh prices every 5 seconds
    const priceInterval = setInterval(fetchPrices, 5000);
    return () => clearInterval(priceInterval);
  }, [data]);

  const handleCancelOrder = async () => {
    if (!cancelConfirm) return;

    setCancelling(true);
    try {
      if (cancelConfirm.type === 'single' && cancelConfirm.orderId) {
        await cancelPendingOrder(cancelConfirm.ticker, cancelConfirm.orderId);
      } else {
        await cancelAllPendingOrders(cancelConfirm.ticker);
      }

      // Refresh orders after cancellation
      await fetchOrders();
      setCancelConfirm(null);
    } catch (err: any) {
      console.error('Failed to cancel order:', err);
      alert(`Failed to cancel: ${err.message || 'Unknown error'}`);
    } finally {
      setCancelling(false);
    }
  };

  // Helper: Calculate distance to current price
  const calculateDistance = (orderPrice: number, currentPrice: number, side: string): {
    distance: number;
    color: string;
    status: 'very-close' | 'close' | 'medium' | 'far';
  } => {
    // For LONG: negative distance means price below current (good)
    // For SHORT: positive distance means price above current (good)
    const rawDistance = ((orderPrice - currentPrice) / currentPrice) * 100;
    const distance = side === 'Long' ? rawDistance : -rawDistance;

    const absDistance = Math.abs(distance);

    let color = 'text-gray-400';
    let status: 'very-close' | 'close' | 'medium' | 'far' = 'far';

    if (absDistance < 0.3) {
      color = 'text-green-400';
      status = 'very-close';
    } else if (absDistance < 0.8) {
      color = 'text-yellow-400';
      status = 'close';
    } else if (absDistance < 2.0) {
      color = 'text-orange-400';
      status = 'medium';
    } else {
      color = 'text-red-400';
      status = 'far';
    }

    return { distance, color, status };
  };

  // Helper: Detect DCA level from order price
  const detectDCALevel = (order: PendingOrder, allOrders: PendingOrder[]): DCAInfo | null => {
    // Get all orders for same ticker and side
    const sameTickerOrders = allOrders.filter(
      o => o.ticker === order.ticker && o.side === order.side
    ).sort((a, b) => {
      // Sort by price: ascending for LONG, descending for SHORT
      return order.side === 'Long' ? b.price - a.price : a.price - b.price;
    });

    if (sameTickerOrders.length < 2) return null; // Not DCA if only 1 order

    const orderIndex = sameTickerOrders.findIndex(o => o.order_id === order.order_id);
    if (orderIndex === -1) return null;

    const level = orderIndex + 1;

    // Guess type based on level
    let type = "Unknown";
    if (level === 1) type = "Fixed offset";
    else if (level === 2) type = "ATR-based";
    else if (level === 3) type = "Swing distance";

    return { level, type };
  };

  // Calculate risk exposure
  const calculateRiskExposure = () => {
    if (!data || data.orders.length === 0) return null;

    let totalValue = 0;
    let totalQuantity: Record<string, number> = {};

    data.orders.forEach(order => {
      const value = order.price * order.quantity;
      totalValue += value;

      if (!totalQuantity[order.ticker]) {
        totalQuantity[order.ticker] = 0;
      }
      totalQuantity[order.ticker] += order.quantity;
    });

    return {
      totalValue,
      totalQuantity,
      orderCount: data.orders.length,
    };
  };

  const riskExposure = calculateRiskExposure();

  // Group orders by ticker
  const ordersByTicker = data?.orders.reduce((acc, order) => {
    if (!acc[order.ticker]) {
      acc[order.ticker] = [];
    }
    acc[order.ticker].push(order);
    return acc;
  }, {} as Record<string, PendingOrder[]>) || {};

  if (error && !data) {
    return (
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold">Pending Orders</h2>
          <button onClick={fetchOrders} className="btn btn-sm btn-primary">
            <RefreshCw size={14} className="mr-1" />
            Retry
          </button>
        </div>
        <div className="p-4 bg-red-900/20 border border-red-500 rounded text-red-300">
          <AlertCircle className="inline mr-2" size={16} />
          {error}
        </div>
      </div>
    );
  }

  if (!data && loading) {
    return (
      <div className="card">
        <h2 className="text-xl font-bold mb-4">Pending Orders</h2>
        <div className="flex items-center justify-center p-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-profit"></div>
          <span className="ml-3 text-dark-text-secondary">Loading orders...</span>
        </div>
      </div>
    );
  }

  if (!data || data.total_count === 0) {
    return (
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold">Pending Orders</h2>
          <button
            onClick={fetchOrders}
            className="btn btn-sm btn-secondary"
            disabled={loading}
          >
            <RefreshCw size={14} className={`mr-1 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
        <div className="p-4 bg-dark-bg border border-dark-border rounded text-dark-text-secondary text-center">
          <Clock className="mx-auto mb-2" size={32} opacity={0.5} />
          <p>No pending orders</p>
          <p className="text-xs mt-1">All limit orders have been filled or cancelled</p>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-xl font-bold">Pending Orders ({data.total_count})</h2>
          <p className="text-xs text-dark-text-secondary mt-1">
            Limit orders waiting for execution
          </p>
        </div>
        <div className="flex items-center gap-2">
          {lastUpdated && (
            <span className="text-xs text-dark-text-secondary">
              Updated {lastUpdated.toLocaleTimeString()}
            </span>
          )}
          <button
            onClick={fetchOrders}
            className="btn btn-sm btn-secondary"
            disabled={loading}
          >
            <RefreshCw size={14} className={`mr-1 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Risk Exposure Summary */}
      {riskExposure && (
        <div className="mb-4 p-4 bg-blue-900/20 border border-blue-500/30 rounded-lg">
          <div className="flex items-center gap-2 mb-3">
            <Shield className="text-blue-400" size={18} />
            <h3 className="font-bold text-blue-300">Risk Exposure Summary</h3>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <div className="text-dark-text-secondary text-xs">Total Pending Value</div>
              <div className="font-mono font-bold text-blue-300 flex items-center gap-1">
                <DollarSign size={14} />
                {riskExposure.totalValue.toFixed(2)}
              </div>
            </div>
            <div>
              <div className="text-dark-text-secondary text-xs">Total Orders</div>
              <div className="font-mono font-bold text-blue-300">
                {riskExposure.orderCount}
              </div>
            </div>
            <div>
              <div className="text-dark-text-secondary text-xs">Positions to Open</div>
              <div className="font-mono font-bold text-blue-300">
                {Object.keys(riskExposure.totalQuantity).length}
              </div>
            </div>
            <div>
              <div className="text-dark-text-secondary text-xs">Capital at Risk</div>
              <div className="font-mono font-bold text-yellow-400">
                {/* Assuming $1000 balance for % calc - adjust as needed */}
                {((riskExposure.totalValue / 1000) * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Orders by Ticker */}
      <div className="space-y-4">
        {Object.entries(ordersByTicker).map(([ticker, orders]) => {
          const currentPrice = currentPrices[ticker];

          return (
            <div key={ticker} className="border border-dark-border rounded-lg p-4 bg-dark-bg">
              {/* Ticker Header */}
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                  <h3 className="text-lg font-bold">{ticker}</h3>
                  {currentPrice && (
                    <span className="text-sm text-dark-text-secondary font-mono">
                      Current: ${currentPrice.toFixed(5)}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <span className="badge badge-info">{orders.length} order{orders.length > 1 ? 's' : ''}</span>
                  <button
                    onClick={() => setCancelConfirm({ type: 'all', ticker })}
                    className="btn btn-sm btn-danger flex items-center gap-1"
                    title="Cancel all orders for this ticker"
                  >
                    <Trash2 size={12} />
                    Cancel All
                  </button>
                </div>
              </div>

              {/* Orders Table */}
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-dark-border text-dark-text-secondary">
                      <th className="text-left py-2">Side</th>
                      <th className="text-left py-2">DCA</th>
                      <th className="text-right py-2">Price</th>
                      <th className="text-right py-2">Distance</th>
                      <th className="text-right py-2">Quantity</th>
                      <th className="text-right py-2">Filled</th>
                      <th className="text-left py-2">Status</th>
                      <th className="text-left py-2">Age</th>
                      <th className="text-left py-2">SL/TP</th>
                      <th className="text-center py-2">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {orders.map((order) => {
                      const fillPercent = order.quantity > 0
                        ? (order.filled_quantity / order.quantity) * 100
                        : 0;

                      const timeSinceCreation = Date.now() - new Date(order.created_at).getTime();
                      const minutesAgo = Math.floor(timeSinceCreation / 1000 / 60);

                      const dcaInfo = detectDCALevel(order, data.orders);

                      let distanceInfo = null;
                      if (currentPrice) {
                        distanceInfo = calculateDistance(order.price, currentPrice, order.side);
                      }

                      return (
                        <tr
                          key={order.order_id}
                          className="border-b border-dark-border/50 hover:bg-dark-card transition-colors"
                        >
                          {/* Side */}
                          <td className="py-3">
                            <span className={`badge badge-sm ${
                              order.side === 'Long'
                                ? 'badge-success'
                                : 'badge-danger'
                            } flex items-center gap-1 w-fit`}>
                              {order.side === 'Long' ? (
                                <TrendingUp size={12} />
                              ) : (
                                <TrendingDown size={12} />
                              )}
                              {order.side}
                            </span>
                          </td>

                          {/* DCA Level */}
                          <td className="py-3">
                            {dcaInfo ? (
                              <div className="flex flex-col">
                                <span className="badge badge-sm bg-purple-600 text-xs">
                                  L{dcaInfo.level}
                                </span>
                                <span className="text-xs text-dark-text-secondary mt-1">
                                  {dcaInfo.type}
                                </span>
                              </div>
                            ) : (
                              <span className="text-dark-text-secondary text-xs">-</span>
                            )}
                          </td>

                          {/* Price */}
                          <td className="py-3 text-right font-mono font-medium">
                            ${order.price.toFixed(5)}
                          </td>

                          {/* Distance */}
                          <td className="py-3 text-right">
                            {distanceInfo ? (
                              <div className="flex flex-col items-end">
                                <span className={`font-mono font-bold ${distanceInfo.color}`}>
                                  {distanceInfo.distance > 0 ? '+' : ''}{distanceInfo.distance.toFixed(2)}%
                                </span>
                                {distanceInfo.status === 'very-close' && (
                                  <span className="text-xs text-green-400 flex items-center gap-1 mt-1">
                                    <Target size={10} />
                                    Very close!
                                  </span>
                                )}
                                {distanceInfo.status === 'far' && (
                                  <span className="text-xs text-red-400 flex items-center gap-1 mt-1">
                                    <AlertTriangle size={10} />
                                    Far away
                                  </span>
                                )}
                              </div>
                            ) : (
                              <span className="text-dark-text-secondary text-xs">-</span>
                            )}
                          </td>

                          {/* Quantity */}
                          <td className="py-3 text-right font-mono">
                            {order.quantity.toFixed(2)}
                          </td>

                          {/* Filled */}
                          <td className="py-3 text-right">
                            {order.filled_quantity > 0 ? (
                              <div>
                                <div className="font-mono text-xs">
                                  {order.filled_quantity.toFixed(2)}
                                </div>
                                <div className="text-xs text-blue-400">
                                  {fillPercent.toFixed(0)}%
                                </div>
                              </div>
                            ) : (
                              <span className="text-dark-text-secondary text-xs">0</span>
                            )}
                          </td>

                          {/* Status */}
                          <td className="py-3">
                            <span className={`badge badge-sm ${
                              order.status === 'New'
                                ? 'badge-warning'
                                : order.status === 'PartiallyFilled'
                                ? 'badge-info'
                                : 'badge-secondary'
                            }`}>
                              {order.status}
                            </span>
                          </td>

                          {/* Age */}
                          <td className="py-3 text-xs text-dark-text-secondary">
                            {minutesAgo < 60
                              ? `${minutesAgo}m ago`
                              : `${Math.floor(minutesAgo / 60)}h ago`
                            }
                          </td>

                          {/* SL/TP */}
                          <td className="py-3 text-xs">
                            {order.stop_loss || order.take_profit ? (
                              <div className="space-y-1">
                                {order.stop_loss && (
                                  <div className="text-loss">
                                    SL: ${order.stop_loss.toFixed(2)}
                                  </div>
                                )}
                                {order.take_profit && (
                                  <div className="text-profit">
                                    TP: ${order.take_profit.toFixed(2)}
                                  </div>
                                )}
                              </div>
                            ) : (
                              <span className="text-dark-text-secondary">-</span>
                            )}
                          </td>

                          {/* Actions */}
                          <td className="py-3 text-center">
                            <button
                              onClick={() => setCancelConfirm({
                                type: 'single',
                                ticker: order.ticker,
                                orderId: order.order_id,
                                orderPrice: order.price
                              })}
                              className="btn btn-sm btn-danger px-2 py-1"
                              title="Cancel this order"
                            >
                              <X size={14} />
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          );
        })}
      </div>

      {/* Auto-refresh indicator */}
      {autoRefresh && (
        <div className="mt-4 text-xs text-dark-text-secondary text-center flex items-center justify-center gap-1">
          <span className="inline-block w-2 h-2 bg-blue-500 rounded-full animate-pulse"></span>
          Auto-refreshing every {refreshInterval / 1000}s
        </div>
      )}

      {/* Cancel Confirmation Modal */}
      {cancelConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-dark-card border border-dark-border rounded-lg p-6 max-w-md w-full mx-4">
            <h3 className="text-xl font-bold mb-4 flex items-center gap-2 text-red-400">
              <AlertTriangle size={24} />
              Confirm Cancellation
            </h3>

            {cancelConfirm.type === 'single' ? (
              <p className="text-dark-text-secondary mb-6">
                Are you sure you want to cancel this order?
                <br />
                <br />
                <span className="font-mono text-sm">
                  {cancelConfirm.ticker} @ ${cancelConfirm.orderPrice?.toFixed(5)}
                </span>
                <br />
                <span className="text-xs">Order ID: {cancelConfirm.orderId?.slice(0, 16)}...</span>
              </p>
            ) : (
              <p className="text-dark-text-secondary mb-6">
                Are you sure you want to cancel <span className="font-bold text-red-400">ALL</span> pending orders for <span className="font-bold">{cancelConfirm.ticker}</span>?
                <br />
                <br />
                <span className="text-xs">This action cannot be undone.</span>
              </p>
            )}

            <div className="flex gap-3">
              <button
                onClick={() => setCancelConfirm(null)}
                className="btn btn-secondary flex-1"
                disabled={cancelling}
              >
                No, Keep Orders
              </button>
              <button
                onClick={handleCancelOrder}
                className="btn btn-danger flex-1"
                disabled={cancelling}
              >
                {cancelling ? 'Cancelling...' : 'Yes, Cancel'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
