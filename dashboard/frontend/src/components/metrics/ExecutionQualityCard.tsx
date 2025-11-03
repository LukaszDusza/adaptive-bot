/**
 * Execution Quality Card - Shows slippage and maker/taker analysis
 */
import React from 'react';
import { TrendingDown, TrendingUp, DollarSign, Zap } from 'lucide-react';
import type { ExecutionQuality } from '../../types';

interface ExecutionQualityCardProps {
  data: ExecutionQuality | null;
  loading?: boolean;
}

export const ExecutionQualityCard: React.FC<ExecutionQualityCardProps> = ({ data, loading }) => {
  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(3)}%`;
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value);
  };

  if (loading || !data) {
    return (
      <div className="stat-card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-dark-text-primary">Execution Quality</h3>
          <Zap className="text-blue-500" size={20} />
        </div>
        <div className="text-center text-dark-text-secondary py-8">
          {loading ? 'Loading...' : 'No data available'}
        </div>
      </div>
    );
  }

  return (
    <div className="stat-card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-dark-text-primary">Execution Quality</h3>
        <Zap className="text-blue-500" size={20} />
      </div>

      <div className="space-y-4">
        {/* Total Executions */}
        <div>
          <div className="text-sm text-dark-text-secondary">Total Executions</div>
          <div className="text-2xl font-bold text-dark-text-primary">
            {data.total_executions}
          </div>
        </div>

        {/* Average Slippage */}
        <div className="border-t border-dark-border pt-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-dark-text-secondary">Avg Slippage</span>
            {data.avg_slippage_pct < 0 ? (
              <TrendingDown className="text-profit" size={16} />
            ) : (
              <TrendingUp className="text-loss" size={16} />
            )}
          </div>
          <div className={`text-xl font-semibold ${data.avg_slippage_pct < 0 ? 'text-profit' : 'text-loss'}`}>
            {formatPercent(data.avg_slippage_pct)}
          </div>
          <div className="text-xs text-dark-text-secondary mt-1">
            Best: {formatPercent(data.best_slippage_pct)} | Worst: {formatPercent(data.worst_slippage_pct)}
          </div>
        </div>

        {/* Maker vs Taker */}
        <div className="border-t border-dark-border pt-3">
          <div className="text-sm text-dark-text-secondary mb-2">Maker vs Taker</div>
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <div className="text-sm font-medium text-profit">Maker {(data.maker_ratio * 100).toFixed(1)}%</div>
              <div className="text-xs text-dark-text-secondary">{data.maker_count} orders</div>
            </div>
            <div className="flex-1">
              <div className="text-sm font-medium text-blue-400">Taker {(data.taker_ratio * 100).toFixed(1)}%</div>
              <div className="text-xs text-dark-text-secondary">{data.taker_count} orders</div>
            </div>
          </div>

          {/* Progress bar */}
          <div className="w-full h-2 bg-dark-bg-secondary rounded-full overflow-hidden mt-2">
            <div
              className="h-full bg-profit"
              style={{ width: `${data.maker_ratio * 100}%` }}
            />
          </div>
        </div>

        {/* Total Fees */}
        <div className="border-t border-dark-border pt-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-dark-text-secondary">Total Fees</span>
            <DollarSign className="text-yellow-500" size={16} />
          </div>
          <div className="text-xl font-semibold text-dark-text-primary">
            {formatCurrency(data.total_fees)}
          </div>
          <div className="text-xs text-dark-text-secondary mt-1">
            Avg per execution: {formatCurrency(data.avg_fee_per_execution)}
          </div>
        </div>
      </div>
    </div>
  );
};
