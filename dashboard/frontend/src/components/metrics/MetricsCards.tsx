/**
 * Main metrics display cards
 */
import React from 'react';
import { TrendingUp, TrendingDown, Target, AlertCircle } from 'lucide-react';
import type { MetricsResponse } from '../../types';

interface MetricsCardsProps {
  metrics: MetricsResponse;
}

export const MetricsCards: React.FC<MetricsCardsProps> = ({ metrics }) => {
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value);
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {/* Closed Trades PnL */}
      <div className="stat-card">
        <div className="flex items-center justify-between">
          <span className="stat-label">Closed Trades P&L</span>
          {metrics.total_pnl >= 0 ? (
            <TrendingUp className="text-profit" size={20} />
          ) : (
            <TrendingDown className="text-loss" size={20} />
          )}
        </div>
        <div className={`stat-value ${metrics.total_pnl >= 0 ? 'text-profit' : 'text-loss'}`}>
          {formatCurrency(metrics.total_pnl)}
        </div>
        <div className="text-sm text-dark-text-secondary">
          From closed positions only
        </div>
      </div>

      {/* Win Rate */}
      <div className="stat-card">
        <div className="flex items-center justify-between">
          <span className="stat-label">Win Rate</span>
          <Target className="text-blue-500" size={20} />
        </div>
        <div className="stat-value">
          {(metrics.win_rate * 100).toFixed(1)}%
        </div>
        <div className="text-sm text-dark-text-secondary">
          {metrics.total_trades} total trades
        </div>
      </div>

      {/* Sharpe Ratio */}
      <div className="stat-card">
        <div className="flex items-center justify-between">
          <span className="stat-label">Sharpe Ratio</span>
          <TrendingUp className="text-purple-500" size={20} />
        </div>
        <div className="stat-value">
          {metrics.sharpe_ratio !== null ? metrics.sharpe_ratio.toFixed(2) : 'N/A'}
        </div>
        <div className="text-sm text-dark-text-secondary">
          Risk-adjusted returns
        </div>
      </div>

      {/* Max Drawdown */}
      <div className="stat-card">
        <div className="flex items-center justify-between">
          <span className="stat-label">Max Drawdown</span>
          <AlertCircle className="text-loss" size={20} />
        </div>
        <div className="stat-value text-loss">
          -{formatCurrency(metrics.max_drawdown)}
        </div>
        <div className="text-sm text-dark-text-secondary">
          -{metrics.max_drawdown_percent.toFixed(2)}% from peak
        </div>
      </div>
    </div>
  );
};
