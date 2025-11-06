/**
 * Trade Detail Modal - Shows full trade details with candlestick chart
 */
import React from 'react';
import { X, TrendingUp, TrendingDown, Clock, DollarSign, Target, Shield } from 'lucide-react';
import { Trade } from '../../types';
import { TradeCandleChart } from '../charts/TradeCandleChart';

interface TradeDetailModalProps {
  trade: Trade;
  onClose: () => void;
}

export const TradeDetailModal: React.FC<TradeDetailModalProps> = ({ trade, onClose }) => {
  // Format helpers
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value);
  };

  const formatPercent = (value: number) => {
    return `${value > 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (hours > 24) {
      const days = Math.floor(hours / 24);
      const remainingHours = hours % 24;
      return `${days}d ${remainingHours}h`;
    }
    return `${hours}h ${minutes}m`;
  };

  const pnl = trade.summary?.pnl || 0;
  const pnlPercent = trade.summary?.pnl_percent || 0;
  const isProfitable = pnl > 0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-75 p-4">
      <div className="bg-dark-bg rounded-lg shadow-2xl max-w-7xl w-full max-h-[95vh] overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-dark-card border-b border-dark-border p-6 flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-dark-text-primary flex items-center gap-3">
              <span className={trade.side === 'Long' ? 'text-profit' : 'text-loss'}>
                {trade.side === 'Long' ? <TrendingUp size={28} /> : <TrendingDown size={28} />}
              </span>
              {trade.ticker}
              <span className={`badge text-lg ${trade.side === 'Long' ? 'badge-success' : 'badge-danger'}`}>
                {trade.side}
              </span>
            </h2>
            <p className="text-sm text-dark-text-secondary mt-1">
              Trade ID: {trade.trade_id}
            </p>
          </div>

          <button
            onClick={onClose}
            className="text-dark-text-secondary hover:text-dark-text-primary transition-colors"
          >
            <X size={28} />
          </button>
        </div>

        {/* Trade Summary Cards */}
        <div className="p-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* P&L Card */}
          <div className="stat-card">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-dark-text-secondary">P&L</span>
              <DollarSign className={isProfitable ? 'text-profit' : 'text-loss'} size={20} />
            </div>
            <div className={`text-2xl font-bold ${isProfitable ? 'text-profit' : 'text-loss'}`}>
              {formatCurrency(pnl)}
            </div>
            <div className={`text-sm ${isProfitable ? 'text-profit' : 'text-loss'}`}>
              {formatPercent(pnlPercent)}
            </div>
          </div>

          {/* Duration Card */}
          <div className="stat-card">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-dark-text-secondary">Duration</span>
              <Clock className="text-blue-400" size={20} />
            </div>
            <div className="text-2xl font-bold text-dark-text-primary">
              {trade.summary ? formatDuration(trade.summary.duration_seconds) : 'Active'}
            </div>
            <div className="text-sm text-dark-text-secondary">
              {new Date(trade.start_time).toLocaleString()}
            </div>
          </div>

          {/* Entry/Exit Card */}
          <div className="stat-card">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-dark-text-secondary">Entry → Exit</span>
              <Target className="text-purple-400" size={20} />
            </div>
            <div className="text-lg font-bold text-dark-text-primary">
              ${trade.entry_price?.toFixed(4) || 'N/A'}
            </div>
            <div className="text-sm text-dark-text-secondary">
              → ${trade.exit_price?.toFixed(4) || 'Active'}
            </div>
          </div>

          {/* Exit Reason Card */}
          <div className="stat-card">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-dark-text-secondary">Exit Reason</span>
              <Shield className="text-orange-400" size={20} />
            </div>
            <div className="text-lg font-bold text-dark-text-primary">
              {trade.summary?.exit_reason || 'Active'}
            </div>
            <div className="text-sm text-dark-text-secondary">
              Quantity: {trade.quantity?.toFixed(4) || 'N/A'}
            </div>
          </div>
        </div>

        {/* Candlestick Chart */}
        <div className="p-6">
          <TradeCandleChart tradeId={trade.trade_id} />
        </div>

        {/* Additional Details */}
        {trade.summary && (
          <div className="p-6 border-t border-dark-border">
            <h3 className="text-lg font-semibold text-dark-text-primary mb-4">Trade Metrics</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {trade.summary.max_favorable_excursion !== null && (
                <div>
                  <span className="text-sm text-dark-text-secondary">Max Favorable Excursion (MFE)</span>
                  <p className="text-lg font-semibold text-profit">
                    {formatPercent(trade.summary.max_favorable_excursion)}
                  </p>
                </div>
              )}
              {trade.summary.max_adverse_excursion !== null && (
                <div>
                  <span className="text-sm text-dark-text-secondary">Max Adverse Excursion (MAE)</span>
                  <p className="text-lg font-semibold text-loss">
                    {formatPercent(trade.summary.max_adverse_excursion)}
                  </p>
                </div>
              )}
              <div>
                <span className="text-sm text-dark-text-secondary">Fees Paid</span>
                <p className="text-lg font-semibold text-dark-text-primary">
                  {formatCurrency(trade.summary.fees_paid)}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Trade Notes */}
        {trade.notes && (
          <div className="p-6 border-t border-dark-border">
            <h3 className="text-lg font-semibold text-dark-text-primary mb-2">Notes</h3>
            <p className="text-dark-text-secondary whitespace-pre-wrap">{trade.notes}</p>
          </div>
        )}

        {/* Close Button */}
        <div className="p-6 border-t border-dark-border flex justify-end">
          <button
            onClick={onClose}
            className="px-6 py-2 bg-dark-card hover:bg-dark-bg-secondary border border-dark-border rounded-md text-dark-text-primary transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
